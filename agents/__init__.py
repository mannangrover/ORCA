"""Agents. Importing this package registers every domain agent.

Two kinds live here, and the difference is the architecture:

* **Language agents** -- ``intent_planner_agent`` and ``narrate`` -- call an
  LLM. One decides what to do, the other decides how to say it. Neither
  produces a number attached to a safety claim.
* **Domain agents** -- weather, geospatial, ocean, risk, discovery -- are deterministic
  Python. They retrieve, compute and threshold. See ``agents/base.py`` for why
  they have no prompt files and never will.

``discovery_agent`` is the odd one: it owns no tools and adds no plan steps.
It reads the *metadata* of everything the others ran -- age, composite rung,
gap-filling -- and judges it against the freshness limits in
``config/risk_thresholds.yaml``. Values belong to the domain agents; only the
provenance of those values belongs to it.

``synthesis_agent`` sits between the two: it assembles the recommendation
object deterministically, and ``narrate`` rewrites the result.

Importing for side effects, exactly as ``tools/__init__.py`` does: each module
calls ``agents.base.register_agent()`` at import time, so ``import agents`` is
what makes ``agents.base.AGENTS`` complete.
"""

from __future__ import annotations

from agents import (  # noqa: F401
    discovery_agent,
    geospatial_agent,
    ocean_agent,
    risk_agent,
    weather_agent,
)
from agents.base import (  # noqa: F401
    AGENTS,
    Agent,
    DomainFragment,
    FragmentFinding,
    agent_for,
    all_agents,
    compose_plan,
    fragments_for,
)

__all__ = [
    "AGENTS",
    "Agent",
    "DomainFragment",
    "FragmentFinding",
    "agent_for",
    "all_agents",
    "compose_plan",
    "fragments_for",
]
