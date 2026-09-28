"""Data records passed between dataset, context strategies, models and evaluation."""
from __future__ import annotations

from pydantic import BaseModel

DECISIONS = [
    "inspect_application",
    "inspect_database",
    "inspect_network",
    "inspect_infrastructure",
    "rollback_deployment",
    "ask_for_more_information",
    "escalate",
]

ASK = "ask_for_more_information"

FactValue = bool | str


class DecisionCase(BaseModel):
    id: str
    request: str
    facts: dict[str, FactValue]      # every known fact, in presentation order
    fact_texts: dict[str, str]       # the same facts rendered as sentences
    relevant_facts: list[str]        # minimal set that determines the decision
    irrelevant_facts: list[str]      # distractors with no bearing on the decision
    correct_decision: str
    requires_clarification: bool = False


class DecisionContext(BaseModel):
    """Everything a model is allowed to see. Never contains the ground truth."""
    request: str
    context_text: str
    available_decisions: list[str]
    facts: dict[str, FactValue] = {}  # structured view of the selected facts (rule baseline)
    fact_texts: list[str] = []        # the selected facts as sentences (backends with own format)


class DecisionResult(BaseModel):
    decision: str
    confidence: float | None = None
    scores: dict[str, float] | None = None  # probability per decision, if the backend has them
    raw_output: object | None = None
    latency_ms: float | None = None


class RunRecord(BaseModel):
    """One line of a results JSONL file."""
    case_id: str
    experiment: str
    model: str
    decision: str
    correct_decision: str
    confidence: float | None = None
    scores: dict[str, float] | None = None
    latency_ms: float
    context_ms: float = 0.0          # time the context strategy took (e.g. a filtering model)
    selected_facts: list[str]
    relevant_facts: list[str]
    raw_output: object | None = None
