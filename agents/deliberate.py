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
* ``concerns`` -- sentences for the caveat block. **Any number in a concern is
  stripped**, because a caveat reading "waves may reach 3 m" would be a safety
  figure produced by a language model. See :func:`strip_numbers`.
* ``assessment`` -- one line for the reasoning trace, same treatment.

So the model may say *"the zone is far offshore for this boat, check the route
home"*. It may not say *how* far, or *how* long. Those come from tools.

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

    @property
    def used_llm(self) -> bool:
        return self.ok and bool(self.model)


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

    completion = llm.complete("deliberator", system, _findings_block(agent, result, intent))
    if not completion.ok:
        return Deliberation(agent=agent.name, error=completion.error or "llm unavailable")

    try:
        assessment, raw_requests, concerns = _parse(completion.text, agent)
    except (ValueError, json.JSONDecodeError) as exc:
        return Deliberation(agent=agent.name, error=f"unparseable: {exc}")

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
                reason=strip_numbers(str(item.get("reason") or "requested by agent")),
                # A model may never mark its own request safety-critical.
                # Critical requests come from the rule floor, which is not
                # subject to a model's judgement on the day.
                critical=False,
            )
        )

    return Deliberation(
        agent=agent.name,
        assessment=strip_numbers(assessment)[:300],
        requests=requests,
        # Filtered AFTER stripping, not before. A concern that is mostly digits
        # -- "waves 3.5 m, gusts 28 kn" -- strips down to punctuation or to
        # nothing at all, and an empty Caveat fails validation and takes the
        # whole recommendation with it. The guard exists to make answers safe;
        # it must not be able to destroy one.
        concerns=[
            stripped
            for c in concerns
            if len(stripped := strip_numbers(str(c))[:200]) >= _MIN_CONCERN_CHARS
        ][:3],
        ok=True,
        model=completion.model,
    )
