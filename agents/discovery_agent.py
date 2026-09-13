"""Marine data discovery: which sources answered, and were they good enough.

The gap this fills is not hypothetical. ``config/risk_thresholds.yaml``
declares a freshness policy -- ``forecast_max_age_h: 12``,
``satellite_max_age_days: 7``, a ``composite_ladder`` of [1, 3, 7] -- and
promises it is *"surfaced in confidence.basis, never silently substituted"*.

Before this module, **nothing in the codebase read any of those keys.** Twelve
files write ``data_age_days`` / ``clear_pass_fraction`` /
``composite_window_days``; two read them, one to format a string and one to
apply a single hardcoded ``> 1.0``. Every tool carefully recorded how stale its
data was and nothing decided anything with it.

So this agent implements a policy the project already wrote down.

---

**Why it does not clash with the domain agents.**

The line is between a value and the metadata about that value:

* ``ocean_agent`` reads ``.concentration``, ``.anomaly_sigma`` -- what the
  water is doing.
* ``discovery_agent`` reads ``.quality`` and ``.provenance`` -- how old that
  reading is, which rung of the composite ladder it came from, whether the
  cell was gap-filled rather than observed.

They read different fields of the same object and never the same one.

**Two prohibitions, so it cannot grow into the other agents' territory:**

1. **It never reads a data value, and never emits one.** Ages, coverage
   fractions and source names only. A number that describes the sea belongs to
   the domain agent that computed it.
2. **It never fetches.** ``ingest/`` owns fetching, on a schedule, and the
   cache-first rule is what makes the system work at a venue with no wifi. This
   agent reads the catalogue and the metadata of what already ran.

**It owns no tools.** Every other domain agent claims a registry group and
contributes plan steps; this one contributes none, because judging the data is
not a step in the plan -- it is a reading of every step after the fact. That is
why ``plan_steps`` returns an empty list unconditionally and why ``fragment``
reads the whole tool log rather than ``_my_records``.

The staleness policy mirrors the split the other agents already use:

* A **stale forecast** is safety-critical. Wave and wind data older than the
  configured limit cannot support a launch decision, and the finding is keyed
  so ``DomainFragment.blocks_verdict`` refuses a clean verdict on it.
* A **stale satellite layer** is not. A week-old chlorophyll field costs a PFZ
  answer its confidence; it does not put anyone to sea in conditions nobody
  checked. This is the same judgement ``ocean_agent`` makes in its own file,
  and it is repeated here on purpose rather than generalised into a shared
  helper that would hide the difference.
"""

from __future__ import annotations

from typing import Any

from agents.base import Agent, AgentRequest, DomainFragment, FragmentFinding, register_agent
from core import config
from core.schemas.intent import Intent
from core.schemas.tool_io import ToolStatus
from ingest.catalogue import resolve_datasets
from tools.registry import AgentGroup

__all__ = ["DiscoveryAgent", "discovery_agent", "SATELLITE_LAYERS", "FORECAST_LAYERS"]

#: Sources whose staleness is a safety matter. These feed the verdict.
FORECAST_LAYERS = {"open_meteo_marine", "open_meteo_forecast", "gdacs_tc"}

#: Sources whose staleness weakens an answer without endangering anyone.
SATELLITE_LAYERS = {
    "jplMURSST41",
    "nesdisVHNnoaaSNPPnoaa20NRTchlaGapfilledDaily",
    "nesdisVHNSQchlaMonthly",
}


class DiscoveryAgent(Agent):
    """Judges the data, never the sea."""

    group = AgentGroup.CATALOGUE
    label = "Discovery"

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def plan_steps(self, intent: Intent, base_id: int = 1) -> list[dict[str, Any]]:
        """Always empty. This agent adds no steps to a plan.

        Judging data quality is not something you do *instead of* fetching
        waves; it is something you do to the waves you fetched. Contributing a
        step here would mean inventing a tool whose only job is to look at
        other tools, which is a worse design than reading the log.
        """
        return []

    # ------------------------------------------------------------------
    # Reading the whole run
    # ------------------------------------------------------------------

    def _limits(self) -> tuple[float, float, list[int]]:
        """Freshness limits from config. Never hardcoded here.

        Thresholds are a deliverable in ``risk_thresholds.yaml`` with a
        provenance tag and a rationale per value. This agent applies them; it
        does not get to pick them, exactly as no agent gets to pick 2.5 m.
        """
        block = config.risk_thresholds().get("data_freshness", {})
        forecast_h = float(block.get("forecast_max_age_h", {}).get("value", 12.0))
        satellite_d = float(block.get("satellite_max_age_days", {}).get("value", 7.0))
        ladder = list(block.get("composite_ladder", [1, 3, 7]))
        return forecast_h, satellite_d, ladder

    def _catalogue(self) -> dict[str, dict[str, Any]]:
        return {record["id"]: record for record in resolve_datasets()}

    def fragment(self, result, intent: Intent) -> DomainFragment:
        """Read every step's provenance and quality; judge each against config.

        Deliberately does not call ``_collect``: that partitions by tool
        ownership through the registry, and this agent owns nothing. It reads
        the whole log because its subject is the run, not a domain.
        """
        forecast_max_h, satellite_max_d, ladder = self._limits()
        catalogue = self._catalogue()

        findings: list[FragmentFinding] = []
        notes: list[str] = []
        sources_seen: set[str] = set()
        step_ids: list[str] = []
        worst = ToolStatus.OK

        for record in sorted(
            result.tool_call_log.values(), key=lambda r: r.tool_call_id
        ):
            step_id = record.step_id
            output = result.outputs.get(step_id) if step_id else None
            if step_id:
                step_ids.append(step_id)
            if output is None or output.status is ToolStatus.FAILED:
                # A failed call has no data to judge. The domain agent that
                # owns it already reports the failure; repeating it here would
                # double-count one problem in the evidence drawer.
                continue

            provenance = output.provenance
            quality = output.quality
            source = provenance.source or "unknown"
            sources_seen.add(source)

            # -- catalogue drift ---------------------------------------
            # A source in use that is not in datasets.yaml means the registry
            # and the code have diverged. Worth catching: the catalogue is what
            # the planner prompt and the evidence drawer both cite.
            if source not in catalogue and source not in ("deterministic", "unknown"):
                findings.append(
                    self._finding(
                        result, step_id, "uncatalogued_source", "discovery.uncatalogued",
                        {"tool": record.tool, "source": source},
                    )
                )
                notes.append(
                    f"{record.tool} reports source {source!r}, which is not in "
                    "datasets.yaml -- the catalogue and the code have drifted"
                )

            age_days = quality.data_age_days
            is_forecast = source in FORECAST_LAYERS
            is_satellite = source in SATELLITE_LAYERS

            # -- staleness ---------------------------------------------
            if age_days is not None:
                age_hours = age_days * 24.0
                if is_forecast and age_hours > forecast_max_h:
                    # Keyed `_failed` so DomainFragment.blocks_verdict picks
                    # it up. A launch decision resting on a forecast we know is
                    # out of date is exactly what the config forbids.
                    findings.append(
                        self._finding(
                            result, step_id, "stale_forecast_failed", "discovery.stale_forecast",
                            {
                                "tool": record.tool,
                                "source": source,
                                "age_hours": round(age_hours, 1),
                                "limit_hours": forecast_max_h,
                            },
                            safety_critical=True,
                        )
                    )
                    worst = ToolStatus.DEGRADED
                elif is_satellite and age_days > satellite_max_d:
                    findings.append(
                        self._finding(
                            result, step_id, "stale_satellite", "discovery.stale_satellite",
                            {
                                "tool": record.tool,
                                "source": source,
                                "age_days": round(age_days, 1),
                                "limit_days": satellite_max_d,
                            },
                            # Not safety-critical. Same policy as ocean_agent:
                            # a missing ocean layer weakens the answer, it does
                            # not endanger the fisherman.
                            safety_critical=False,
                        )
                    )
                    worst = ToolStatus.DEGRADED

            # -- which rung of the composite ladder --------------------
            rung = quality.composite_window_days
            if rung is not None and rung > 1:
                findings.append(
                    self._finding(
                        result, step_id, "composite_rung", "discovery.composite",
                        {
                            "tool": record.tool,
                            "window_days": rung,
                            "ladder": ladder,
                            "is_last_rung": rung >= max(ladder) if ladder else False,
                        },
                    )
                )

            # -- interpolated rather than observed ---------------------
            if quality.gap_filled:
                findings.append(
                    self._finding(
                        result, step_id, "gap_filled", "discovery.gap_filled",
                        {
                            "tool": record.tool,
                            "source": source,
                            "coverage": quality.coverage_fraction,
                            "clear_pass_fraction": quality.clear_pass_fraction,
                        },
                    )
                )

        # Silence on a clean run. `fragments_for` skips a fragment with no
        # findings and no steps, and that is the behaviour we want: a caveat
        # attached to every answer teaches people to ignore caveats, and then
        # the one that matters goes unread. An inventory of sources that are
        # all fresh is not news.
        #
        # This also keeps the promise `fragments_for` makes -- no hollow
        # blocks -- which a fragment emitted unconditionally would have broken
        # for every query type.
        if not findings:
            return DomainFragment(agent=self.name, status=ToolStatus.OK)

        findings.append(
            FragmentFinding(
                key="source_inventory",
                template="discovery.inventory",
                slots={
                    "sources": sorted(sources_seen),
                    "count": len(sources_seen),
                    "operator_maintained": sorted(
                        s for s in sources_seen
                        if catalogue.get(s, {}).get("status") == "operator"
                    ),
                },
            )
        )

        return DomainFragment(
            agent=self.name,
            status=worst,
            findings=findings,
            step_ids=step_ids,
            failed_steps=[],
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Collaboration
    # ------------------------------------------------------------------

    def review(self, result, intent: Intent) -> list[AgentRequest]:
        """No requests, and the reason is worth stating.

        Every other agent's ``review`` asks a peer for a tool call it cannot
        make itself. Discovery has nothing to ask for: when a layer is stale
        the remedy is an ingest run on a schedule, not another tool call in
        this turn, and re-requesting the same tool would return the same cached
        file with the same timestamp.

        Returning an empty list is the honest answer. A request that cannot
        change the outcome is theatre, and this architecture bounds
        deliberation rounds precisely so agents do not spend them on it.
        """
        return []

    # ------------------------------------------------------------------
    # The sentence synthesis needs
    # ------------------------------------------------------------------

    def basis(self, fragment: DomainFragment) -> str:
        """One line for ``confidence.basis``, built from the findings.

        This is what the config promised and never delivered. It replaces a
        machine-shaped join of tool and source strings with a statement about
        whether the data supports the answer.
        """
        stale_forecast = [f for f in fragment.findings if f.key == "stale_forecast_failed"]
        stale_satellite = [f for f in fragment.findings if f.key == "stale_satellite"]
        composites = [f for f in fragment.findings if f.key == "composite_rung"]
        gap_filled = [f for f in fragment.findings if f.key == "gap_filled"]
        inventory = next(
            (f for f in fragment.findings if f.key == "source_inventory"), None
        )

        parts: list[str] = []
        if inventory is not None:
            parts.append(f"{inventory.slots['count']} source(s)")

        for finding in stale_forecast:
            parts.append(
                f"{finding.slots['tool']} forecast is {finding.slots['age_hours']} h old, "
                f"past the {finding.slots['limit_hours']} h limit -- too old to support a verdict"
            )
        for finding in stale_satellite:
            parts.append(
                f"{finding.slots['tool']} is {finding.slots['age_days']} days old, "
                f"past the {finding.slots['limit_days']} day limit"
            )
        for finding in composites:
            rung = finding.slots["window_days"]
            tail = " (last rung on the ladder)" if finding.slots.get("is_last_rung") else ""
            parts.append(f"{finding.slots['tool']} fell back to a {rung}-day composite{tail}")
        if gap_filled:
            tools = ", ".join(sorted({f.slots["tool"] for f in gap_filled}))
            parts.append(f"gap-filled (interpolated, not observed): {tools}")

        if inventory is not None and inventory.slots.get("operator_maintained"):
            names = ", ".join(inventory.slots["operator_maintained"])
            parts.append(f"operator-maintained, not an official feed: {names}")

        if not stale_forecast and not stale_satellite and not composites:
            parts.append("all sources within their configured freshness limits")

        return "; ".join(parts) + "."


discovery_agent = register_agent(DiscoveryAgent())
