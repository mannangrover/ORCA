"""Domain agents reasoning about their own results.

**This is the third and last place an LLM touches a turn**, after the planner
and the narrator. It exists because "collaborative agents" should mean agents
that work out what they need, not agents that follow rules somebody typed.

Before this, an agent's decision to ask another agent for help was a hardcoded
condition: *if a fishing zone was found, ask for a boundary check*. Two rules,
covering two situations I happened to think of. An agent that met a situation I
had not anticipated did nothing.

Now each domain agent gets shown its own findings and the tool catalogue, and
decides for itself what is missing.

---------------------------------------------------------------------------
THE LINE THIS DOES NOT CROSS
---------------------------------------------------------------------------

CLAUDE.md's governing rule is unchanged and is what makes this safe:

    The LLM interprets, plans, asks, hypothesises and narrates.
    Code retrieves, computes, thresholds and verifies.

A deliberating agent **decides what to do**. It never decides what is true.
Concretely, its output is:

* ``requests`` -- a tool to run and arguments to run it with. Every one is
  turned into a plan step and put through the same ``validate_plan()`` as
  anything else, so it cannot invent a tool, mis-type an argument, or mark its
  own request safety-critical.
* ``concerns`` -- sentences for the caveat block. **Every number in a concern
  is checked against the tool call log, and the concern is dropped if a figure
  is not there.** See :mod:`agents.grounding` for why checking beat the
  earlier approach of deleting digits outright.
* ``assessment`` -- one line for the reasoning trace, same treatment.

So the model may say *"the swell is the binding driver at 2.2 m against the
2.5 m limit"* when the tools returned those numbers, and may not say it when
they did not. It is allowed to be exactly as specific as its evidence, which
is a stronger position than the old rule of never being specific at all.

---------------------------------------------------------------------------
THE HARDCODED FLOOR STAYS
---------------------------------------------------------------------------

Each agent's rule-based ``review()`` still runs, and its requests are merged
with the model's. The model can **add**, never remove. Forgetting to geofence a
candidate zone means a fisherman arrested in Sri Lankan waters, and that check
is not left to a model's judgement on the day.

Deliberation is also entirely optional: no key, a timeout, a rate limit or
unparseable output all fall back to the rules alone, and the turn proceeds.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agents.grounding import grounded
from core import config
from orchestrator.llm import client as llm
from tools import registry

if TYPE_CHECKING:  # pragma: no cover
    from agents.base import Agent, AgentRequest
    from core.schemas.intent import Intent
    from orchestrator.executor import ExecutionResult

__all__ = ["Deliberation", "deliberate", "strip_numbers", "PROMPT_DIR"]

PROMPT_DIR = Path(__file__).parent / "prompts"

#: Digit runs, including decimals and ranges. Used to strip figures from model
#: prose before it is shown.
_NUMBER = re.compile(r"\d[\d,]*\.?\d*")

#: A deliberating agent may not ask for more than this in one round. A model
#: that returns eleven requests has not prioritised; it has listed.
MAX_REQUESTS = 3

#: Below this a stripped concern is not a sentence. Anything shorter is what
#: remains of a number after the guard ran, and shows the user nothing.
_MIN_CONCERN_CHARS = 12


@dataclass
class Deliberation:
    """What one agent concluded about its own results."""

    agent: str
    assessment: str = ""
    requests: list["AgentRequest"] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    ok: bool = False
    error: str | None = None
    model: str = ""
    #: Figures this agent wrote that no tool returned, and the fragments they
    #: cost. Empty on the overwhelming majority of turns. Non-empty is worth
    #: reading: it is a model reaching for a number it does not have, which
    #: used to be silently erased mid-sentence and is now countable.
    rejected: list[str] = field(default_factory=list)

    @property
    def used_llm(self) -> bool:
        return self.ok and bool(self.model)


def _shown_numbers(findings_json: str) -> list[float]:
    """Every number in the block this agent was actually handed.

    **Not** every number the turn produced. Scoping to the whole tool call
    log looked equivalent and was not: it let an agent quote a figure from a
    domain it never saw, and worse, it let one quote another tool's internal
    intermediates. Observed live on the first run -- the risk agent, whose
    fragment carries normalised sub-scores, reported *"wind speed is 0.0666"*.
    That number was real, it was in the log, and it is a weighting, not a
    wind speed. A grounding check cannot catch a mislabelled number; the only
    defence is to narrow what an agent can reach for.

    So the contract is the simplest one available, and the one a reader can
    hold in their head: **you may quote back what you were shown.**
    """
    pool: list[float] = []

    def walk(node: Any) -> None:
        if isinstance(node, bool):
            return
        if isinstance(node, (int, float)):
            pool.append(float(node))
        elif isinstance(node, str):
            # Figures live inside strings too: the vessel class arrives as
            # "frp_9m", and an agent describing "a 9 m FRP boat" is quoting
            # what it was told, not inventing a hull.
            for token in _NUMBER.findall(node):
                try:
                    pool.append(float(token.replace(",", "")))
                except ValueError:
                    continue
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    try:
        walk(json.loads(findings_json))
    except (ValueError, json.JSONDecodeError):
        pass
    return pool + _threshold_numbers()


def _threshold_numbers() -> list[float]:
    """Every limit in ``config/risk_thresholds.yaml``.

    Always quotable, by any agent, whether or not its own fragment happened
    to carry them. These are not model output and not tool output -- they are
    hand-authored config with a citation per entry, which is precisely the
    file we open when a judge asks why 2.5 m. An agent saying "under the
    2.5 m limit" is reading the rulebook, and the geospatial agent knowing
    the limit without having run the wave tool is correct, not a leak.
    """
    global _THRESHOLDS_CACHE
    if _THRESHOLDS_CACHE is None:
        found: list[float] = []

        def walk(node: Any) -> None:
            if isinstance(node, bool):
                return
            if isinstance(node, (int, float)):
                found.append(float(node))
            elif isinstance(node, dict):
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        try:
            walk(config.load_yaml("risk_thresholds.yaml"))
        except Exception:  # noqa: BLE001 -- a missing config narrows the pool, never breaks a turn
            found = []
        _THRESHOLDS_CACHE = found
    return _THRESHOLDS_CACHE


#: Parsed once. The file is a deliverable that changes between releases, not
#: between turns.
_THRESHOLDS_CACHE: list[float] | None = None


def _ground(text: str, pool: list[float], rejected: list[str]) -> str:
    """The fragment if every number in it is real, else the empty string.

    Records what was thrown away. The three call sites all already had to
    handle "the model said nothing useful here", so rejection reuses that
    path rather than introducing a second failure mode.
    """
    kept, bad = grounded(text.strip(), pool)
    rejected.extend(bad)
    return kept


def strip_numbers(text: str) -> str:
    """Remove numeric tokens from model prose.

    Blunt on purpose. A domain agent's words reach the user as caveats, and a
    caveat is exactly where an invented figure would look most authoritative:
    "conditions may exceed 3 m on the way back" reads like a forecast and is
    not one. The tool outputs carry every real number, and synthesis puts them
    in the claims where the verifier can check them.

    "within 20 km" becomes "within  km", which is ugly. That is the intended
    trade: a clumsy sentence is recoverable, a fabricated safety number is not.
    Agents are told in the prompt not to write numbers, so this should rarely
    fire -- it is the guard, not the mechanism.
    """
    return _NUMBER.sub("", text).replace("  ", " ").strip()


#: Agent group -> prompt filename. The two differ for geospatial, whose prompt
#: CLAUDE.md names ``geo.md``. Without this mapping the geospatial agent
#: silently never deliberated -- it looked for ``geospatial.md``, found nothing,
#: and reported "no prompt file" rather than failing.
_PROMPT_FILES = {
    "ocean": "ocean.md",
    "weather": "weather.md",
    "geospatial": "geo.md",
    "risk": "risk.md",
    "catalogue": "discovery.md",
}


def _prompt_for(agent: "Agent") -> str | None:
    """The agent's prompt file, or None if it has none.

    Prompts live as ``.md`` files per CLAUDE.md -- never inline strings -- and
    the tool catalogue is injected from the registry so it cannot drift.
    """
    path = PROMPT_DIR / _PROMPT_FILES.get(agent.group, f"{agent.group}.md")
    if not path.exists():
        return None
    template = path.read_text(encoding="utf-8").strip()
    if not template:
        return None
    # Compact, and without this agent's own tools: it has already run those,
    # and the full planner catalogue is 5 kB of identical text per agent per
    # turn, which is what exhausted the free-tier token budget.
    return template.replace(
        "{TOOLS}", registry.compact_tool_block(exclude_agent=agent.group)
    ).replace("{AGENT}", agent.label)


def _findings_block(agent: "Agent", result: "ExecutionResult", intent: "Intent") -> str:
    """What this agent found, as JSON, for the model to read.

    The fragment, not the raw tool output: it is already the typed summary this
    domain produced, and handing over a full 24-hour timeseries would bury the
    question in numbers the model has no business reasoning about anyway.
    """
    fragment = agent.fragment(result, intent)
    payload = {
        "query_type": intent.query_type.value,
        "vessel_class": intent.vessel_class.value if intent.vessel_class else None,
        "place": intent.spatial_reference.name if intent.spatial_reference else None,
        "status": fragment.status.value,
        "findings": [
            {"key": f.key, "slots": f.slots, "safety_critical": f.safety_critical}
            for f in fragment.findings
        ],
        "failed_steps": fragment.failed_steps,
        "notes": fragment.notes,
        "tools_already_run": sorted(
            {record.tool for record in result.tool_call_log.values()}
        ),
    }
    return json.dumps(payload, default=str, indent=1)


def _parse(raw: str, agent: "Agent") -> tuple[str, list[dict[str, Any]], list[str]]:
    """Pull the JSON object out of a model reply.

    Models wrap JSON in prose and code fences no matter what the prompt says,
    so the first ``{`` to the last ``}`` is taken rather than trusting the
    whole reply to parse.
    """
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in reply")
    payload = json.loads(raw[start : end + 1])
    assessment = str(payload.get("assessment") or "")
    requests = payload.get("requests") or []
    concerns = payload.get("concerns") or []
    if not isinstance(requests, list) or not isinstance(concerns, list):
        raise ValueError("requests and concerns must be lists")
    return assessment, requests, [str(c) for c in concerns]


def deliberate(
    agent: "Agent", result: "ExecutionResult", intent: "Intent"
) -> Deliberation:
    """Let one agent reason about its results. Never raises.

    Returns an empty :class:`Deliberation` with ``ok=False`` whenever the model
    is unavailable or its reply cannot be used, which the caller treats as "this
    agent had nothing to add" -- the rule-based requests still stand.
    """
    from agents.base import AgentRequest

    system = _prompt_for(agent)
    if system is None:
        return Deliberation(agent=agent.name, error="no prompt file")

    shown = _findings_block(agent, result, intent)
    completion = llm.complete("deliberator", system, shown)
    if not completion.ok:
        return Deliberation(agent=agent.name, error=completion.error or "llm unavailable")

    try:
        assessment, raw_requests, concerns = _parse(completion.text, agent)
    except (ValueError, json.JSONDecodeError) as exc:
        return Deliberation(agent=agent.name, error=f"unparseable: {exc}")

    pool = _shown_numbers(shown)
    rejected: list[str] = []

    requests: list[AgentRequest] = []
    for item in raw_requests[:MAX_REQUESTS]:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or "")
        # Refuse unknown tools here as well as in validate_plan, so the trace
        # records that the model invented one rather than showing a bare
        # validation error later.
        if not registry.has(tool) or not registry.get(tool).implemented:
            continue
        args = item.get("args")
        if not isinstance(args, dict):
            continue
        requests.append(
            AgentRequest(
                from_agent=agent.name,
                to_agent=str(item.get("to_agent") or registry.get(tool).agent),
                tool=tool,
                args=args,
                # A reason with an invented figure loses the reason, never
                # the request: the request carries structured arguments that
                # validate_plan checks far more strictly than any prose, and
                # dropping a safety-relevant tool call because its rationale
                # was badly worded would be the guard doing harm.
                reason=_ground(
                    str(item.get("reason") or ""), pool, rejected
                ) or "requested by agent",
                # A model may never mark its own request safety-critical.
                # Critical requests come from the rule floor, which is not
                # subject to a model's judgement on the day.
                critical=False,
            )
        )

    # A concern is kept whole or not at all. Under the old stripping guard a
    # concern that was mostly digits -- "waves 3.5 m, gusts 28 kn" -- eroded
    # to punctuation, and an empty Caveat fails validation and took the whole
    # recommendation with it; the length floor existed to catch that wreckage.
    # Nothing erodes now, so the floor only has to reject a genuinely empty
    # string, but it is kept: a model can still return " ".
    kept_concerns = [
        text
        for c in concerns
        if len(text := _ground(str(c)[:200], pool, rejected)) >= _MIN_CONCERN_CHARS
    ][:3]

    return Deliberation(
        agent=agent.name,
        assessment=_ground(assessment[:300], pool, rejected),
        requests=requests,
        concerns=kept_concerns,
        ok=True,
        model=completion.model,
        rejected=rejected,
    )
