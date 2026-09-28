"""Turns a case plus selected fact keys into the one standard prompt every model receives."""
from __future__ import annotations

from context_decisions.schemas import DECISIONS, DecisionCase, DecisionContext


def build_context(case: DecisionCase, selected: list[str]) -> DecisionContext:
    facts = "\n".join(f"- {case.fact_texts[k]}" for k in selected) or "- (none)"
    options = "\n".join(f"- {d}" for d in DECISIONS)
    text = (
        "TASK:\nSelect the most appropriate next action.\n\n"
        f"REQUEST:\n{case.request}\n\n"
        f"KNOWN FACTS:\n{facts}\n\n"
        f"OPTIONS:\n{options}\n\n"
        "Return exactly one decision."
    )
    return DecisionContext(
        request=case.request,
        context_text=text,
        available_decisions=list(DECISIONS),
        facts={k: case.facts[k] for k in selected},
    )
