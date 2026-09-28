"""Deterministic baseline: the generator's own rules applied to the facts the model was shown."""
from __future__ import annotations

from context_decisions import domain
from context_decisions.schemas import DecisionContext, DecisionResult


class RuleDecisionModel:
    name = "rules"

    def decide(self, context: DecisionContext) -> DecisionResult:
        decision, reason = domain.decide_with_reason(context.facts)
        return DecisionResult(decision=decision, confidence=1.0, raw_output={"rule_facts": reason})
