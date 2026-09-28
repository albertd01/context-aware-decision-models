from __future__ import annotations

from typing import Protocol

from context_decisions.schemas import DecisionContext, DecisionResult


class DecisionModel(Protocol):
    name: str

    def decide(self, context: DecisionContext) -> DecisionResult: ...
