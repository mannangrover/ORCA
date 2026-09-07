"""Discovery agent: does the data support the answer being given?

Offline, like the other agent tests: hand-built ``ExecutionResult`` objects, no
cache and no network. The point of this agent is a judgement about metadata, so
metadata is all these tests need to construct.

The tests are organised around the two properties that keep the agent honest:
it judges **only** provenance and quality, never a value; and it applies the
limits in ``config/risk_thresholds.yaml`` rather than any of its own.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

import agents
from agents.base import all_agents, fragments_for
from agents.discovery_agent import FORECAST_LAYERS, SATELLITE_LAYERS, DiscoveryAgent
from core import config
from core.provenance import ToolCallLog
from core.schemas.intent import Intent, QueryType, SpatialReference, VesselClass
from core.schemas.tool_io import DataQuality, Provenance, ToolStatus
from core.units import Range, Unit
from orchestrator.executor import ExecutionResult
from tools.ocean.chl_anomaly import ChlAnomalyOut
from tools.weather.wave_forecast import WaveForecastOut

AGENT = DiscoveryAgent()


def _intent(query_type: QueryType = QueryType.SAFETY_ASSESS) -> Intent:
    return Intent(
        query_type=query_type,
        raw_query="q",
        spatial_reference=SpatialReference(name="Nagapattinam"),
        vessel_class=VesselClass.FRP_9M,
    )


def _wave(quality: DataQuality | None = None, source: str = "open_meteo_marine"):
    return WaveForecastOut(
        provenance=Provenance(source=source, authority="Open-Meteo"),
        quality=quality or DataQuality(),
        significant_wave_height=Range(min=0.37, max=0.53, unit=Unit.METRE),
        wave_period=Range(min=4.13, max=6.29, unit=Unit.SECOND),
    )


def _chl(quality: DataQuality | None = None):
    return ChlAnomalyOut(
        provenance=Provenance(
            source="nesdisVHNnoaaSNPPnoaa20NRTchlaGapfilledDaily",
            authority="NOAA CoastWatch",
        ),
        quality=quality or DataQuality(),
        concentration=Range(min=1.37, max=2.94, unit=Unit.MILLIGRAM_PER_CUBIC_METRE),
    )


def _result(*pairs) -> ExecutionResult:
    """Build an ExecutionResult from (step_id, tool, output) triples."""
    log = ToolCallLog(turn_id="t_test")
    outputs, call_ids = {}, {}
    for step_id, tool, output in pairs:
        record = log.record(
            tool=tool,
            step_id=step_id,
            args={},
            output=output,
            started_at=datetime.now(timezone.utc),
            duration_ms=1,
        )
        outputs[step_id] = output
        call_ids[step_id] = record.tool_call_id
    return ExecutionResult(log=log, outputs=outputs, call_ids=call_ids)


def _keys(fragment) -> set[str]:
    return {f.key for f in fragment.findings}


# ==========================================================================
# It owns nothing and plans nothing
# ==========================================================================


def test_it_owns_no_tools():
    """Every other domain agent claims a registry group with tools in it.

    This one judges what the others ran, so owning a tool would mean inventing
    one whose only job is to look at other tools.
    """
    assert AGENT.tools() == []


def test_it_contributes_no_plan_steps():
    for query_type in QueryType:
        assert AGENT.plan_steps(_intent(query_type)) == []


def test_it_is_registered_as_an_agent():
    assert any(a.name == "DiscoveryAgent" for a in all_agents())


# ==========================================================================
# Silence when there is nothing to say
# ==========================================================================


def test_a_clean_run_produces_no_fragment_at_all():
    """A caveat on every answer teaches people to ignore caveats.

    Fresh data inside every limit is not news, and `fragments_for` drops a
    fragment with no findings and no steps -- which is also what keeps a
    geofence answer from carrying a hollow discovery block.
    """
    result = _result(("s2", "wave_forecast", _wave()))
    fragment = AGENT.fragment(result, _intent())
    assert fragment.findings == []
    assert fragment.step_ids == []
    assert "DiscoveryAgent" not in {f.agent for f in fragments_for(result, _intent())}


def test_silence_also_means_no_deliberation_call():
    """The live loop gates deliberation on `fragment().step_ids`.

    An empty fragment therefore costs no LLM call, which matters on a free
    tier where six agents deliberating exhausts the minute budget.
    """
    result = _result(("s2", "wave_forecast", _wave()))
    assert not AGENT.fragment(result, _intent()).step_ids


def test_a_failed_call_is_not_double_reported():
    """The domain agent that owns a failed tool already reports the failure.

    Repeating it here would show one problem twice in the evidence drawer.
    """
    broken = _wave()
    broken = broken.model_copy(update={"status": ToolStatus.FAILED, "error": "timeout"})
    fragment = AGENT.fragment(_result(("s2", "wave_forecast", broken)), _intent())
    assert fragment.findings == []


# ==========================================================================
# Staleness, and the safety split
# ==========================================================================


def test_a_stale_forecast_is_safety_critical_and_blocks_a_verdict():
    """A launch decision resting on out-of-date wind and wave data is the one
    staleness case here that is genuinely dangerous."""
    limit_h = config.risk_thresholds()["data_freshness"]["forecast_max_age_h"]["value"]
    stale = DataQuality(data_age_days=(limit_h * 3) / 24.0)

    fragment = AGENT.fragment(_result(("s2", "wave_forecast", _wave(stale))), _intent())

    assert "stale_forecast_failed" in _keys(fragment)
    assert fragment.blocks_verdict, "a stale forecast must stop a clean verdict"
    assert fragment.status is ToolStatus.DEGRADED


def test_a_stale_satellite_layer_is_not_safety_critical():
    """Same policy the ocean agent applies in its own file: a missing ocean
    layer costs the answer its fishing advice, it does not put anyone to sea
    in conditions nobody checked."""
    limit_d = config.risk_thresholds()["data_freshness"]["satellite_max_age_days"]["value"]
    stale = DataQuality(data_age_days=limit_d + 3)

    fragment = AGENT.fragment(_result(("s2", "chl_anomaly", _chl(stale))), _intent())

    assert "stale_satellite" in _keys(fragment)
    assert not fragment.blocks_verdict, "stale chlorophyll must not block a verdict"


def test_data_inside_the_limits_raises_nothing():
    limit_h = config.risk_thresholds()["data_freshness"]["forecast_max_age_h"]["value"]
    fresh = DataQuality(data_age_days=(limit_h / 2) / 24.0)
    fragment = AGENT.fragment(_result(("s2", "wave_forecast", _wave(fresh))), _intent())
    assert "stale_forecast_failed" not in _keys(fragment)


def test_the_limits_come_from_config_not_from_the_code(monkeypatch):
    """Thresholds are a deliverable with a rationale per value, exactly like
    the 2.5 m wave limit. This agent applies them; it does not choose them.

    Proven by moving the config value and watching the verdict change.
    """
    original = config.risk_thresholds()

    tightened = {
        **original,
        "data_freshness": {
            **original["data_freshness"],
            "forecast_max_age_h": {"value": 0.001},
        },
    }
    monkeypatch.setattr(config, "risk_thresholds", lambda: tightened)

    quality = DataQuality(data_age_days=1.0 / 24.0)  # one hour old
    fragment = AGENT.fragment(_result(("s2", "wave_forecast", _wave(quality))), _intent())
    assert "stale_forecast_failed" in _keys(fragment), (
        "tightening the config limit must change the judgement"
    )


# ==========================================================================
# The rest of the metadata
# ==========================================================================


def test_a_composite_fallback_is_reported_with_its_rung():
    """Interpolating over a week of cloud is legitimate and is not the same as
    seeing the water. Which rung was used is recorded, per the config note."""
    quality = DataQuality(composite_window_days=7, data_age_days=1.0)
    fragment = AGENT.fragment(_result(("s2", "chl_anomaly", _chl(quality))), _intent())

    composite = next(f for f in fragment.findings if f.key == "composite_rung")
    assert composite.slots["window_days"] == 7
    assert composite.slots["is_last_rung"] is True


def test_gap_filled_data_is_flagged_as_computed_not_observed():
    quality = DataQuality(gap_filled=True, coverage_fraction=0.62)
    fragment = AGENT.fragment(_result(("s2", "chl_anomaly", _chl(quality))), _intent())
    assert "gap_filled" in _keys(fragment)


def test_a_source_missing_from_the_catalogue_is_caught():
    """Catalogue drift is worth catching: `datasets.yaml` is what the planner
    prompt and the evidence drawer both cite, so a source in use that is not
    listed means the two have diverged."""
    rogue = _wave(DataQuality(gap_filled=True), source="some_undocumented_feed")
    fragment = AGENT.fragment(_result(("s2", "wave_forecast", rogue)), _intent())

    assert "uncatalogued_source" in _keys(fragment)
    assert any("datasets.yaml" in note for note in fragment.notes)


def test_an_operator_maintained_source_is_named_in_the_inventory():
    """Somebody typed it in by hand. The user is entitled to know that."""
    stale = DataQuality(data_age_days=99.0)
    result = _result(("s2", "wave_forecast", _wave(stale)))
    fragment = AGENT.fragment(result, _intent())
    inventory = next(f for f in fragment.findings if f.key == "source_inventory")
    assert "sources" in inventory.slots
    assert inventory.slots["count"] >= 1


# ==========================================================================
# The boundary: metadata only, never a value
# ==========================================================================


def test_it_never_emits_a_value_from_the_sea():
    """The clash guard, asserted.

    `ocean_agent` reads `.concentration`; this agent reads `.quality`. If a
    wave height or a chlorophyll concentration ever appears in a discovery
    finding, the two agents have started overlapping and one of them is
    redundant.
    """
    quality = DataQuality(data_age_days=99.0, gap_filled=True, composite_window_days=7)
    result = _result(
        ("s2", "wave_forecast", _wave(quality)),
        ("s3", "chl_anomaly", _chl(quality)),
    )
    fragment = AGENT.fragment(result, _intent())

    forbidden = {"significant_wave_height", "wave_period", "concentration", "anomaly_sigma"}
    for finding in fragment.findings:
        assert not (forbidden & set(finding.slots)), f"{finding.key} leaked a sea value"
    # The fixtures use deliberately odd values -- 0.37/0.53 m and
    # 1.37/2.94 mg/m3 -- so that a legitimate integer like a source count
    # can never be mistaken for a leaked measurement. An earlier version
    # used 2.0 for chlorophyll and tripped on `count: 2`.
    leaked = {0.37, 0.53, 4.13, 6.29, 1.37, 2.94}
    for finding in fragment.findings:
        present = {v for v in finding.slots.values() if isinstance(v, (int, float))}
        assert not (leaked & present), f"{finding.key} leaked a sea value"


def test_it_asks_no_other_agent_for_anything():
    """When a layer is stale the remedy is an ingest run on a schedule, not
    another tool call this turn -- which would read the same cached file with
    the same timestamp. An empty list is the honest answer."""
    stale = DataQuality(data_age_days=99.0)
    result = _result(("s2", "wave_forecast", _wave(stale)))
    assert AGENT.review(result, _intent()) == []


# ==========================================================================
# What synthesis gets
# ==========================================================================


def test_basis_states_the_problem_in_words():
    """`confidence.basis` was a machine-shaped join of tool and source strings.
    The config promised freshness would be "surfaced in confidence.basis"; this
    is that promise kept."""
    limit_h = config.risk_thresholds()["data_freshness"]["forecast_max_age_h"]["value"]
    stale = DataQuality(data_age_days=(limit_h * 3) / 24.0)
    fragment = AGENT.fragment(_result(("s2", "wave_forecast", _wave(stale))), _intent())

    basis = AGENT.basis(fragment)
    assert "too old to support a verdict" in basis
    assert basis.endswith(".")


def test_basis_on_a_clean_run_says_so_plainly():
    fragment = AGENT.fragment(_result(("s2", "wave_forecast", _wave())), _intent())
    assert "within their configured freshness limits" in AGENT.basis(fragment)


# ==========================================================================
# Prompt
# ==========================================================================


def test_it_has_a_deliberation_prompt_that_forbids_numbers():
    from agents.deliberate import _prompt_for

    prompt = _prompt_for(AGENT)
    assert prompt, "discovery must have a deliberation prompt"
    assert "never write a number" in prompt.lower()
    assert "{TOOLS}" not in prompt, "the tool catalogue must be injected"


def test_the_prompt_states_the_boundary_against_the_domain_agents():
    from agents.deliberate import _prompt_for

    prompt = _prompt_for(AGENT).lower()
    assert "the data, never the sea" in prompt


def test_the_layer_sets_are_disjoint():
    """A source cannot be both a forecast and a satellite layer; if it were,
    whichever branch ran first would decide whether staleness is dangerous."""
    assert not (FORECAST_LAYERS & SATELLITE_LAYERS)
