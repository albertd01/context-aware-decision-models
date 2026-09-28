"""Context strategies decide *which* facts a model sees. They return fact keys only;
turning them into a prompt is the builder's job."""
from __future__ import annotations

from context_decisions.context.retriever import TfidfRetriever
from context_decisions.domain import RETRIEVAL_QUERY_HINTS
from context_decisions.schemas import DecisionCase


class NoContext:
    name = "no_context"

    def select(self, case: DecisionCase) -> list[str]:
        return []


class FullContext:
    name = "full_context"

    def select(self, case: DecisionCase) -> list[str]:
        return list(case.facts)


class OracleContext:
    """Exactly the relevant facts: an upper bound for any retriever."""
    name = "oracle_context"

    def select(self, case: DecisionCase) -> list[str]:
        return [k for k in case.facts if k in case.relevant_facts]


class RetrievedContext:
    """Top-k facts for the request. An optional `query_suffix` adds domain vocabulary to the
    query, since vague symptom reports share few words with the facts that explain them."""

    def __init__(self, retriever: TfidfRetriever, k: int, query_suffix: str = ""):
        self.retriever = retriever
        self.k = k
        self.query_suffix = query_suffix
        self.name = f"retrieval_expanded_k{k}" if query_suffix else f"retrieval_k{k}"

    def select(self, case: DecisionCase) -> list[str]:
        key_by_text = {text: key for key, text in case.fact_texts.items()}
        query = f"{case.request} {self.query_suffix}".strip()
        texts = self.retriever.retrieve(query, list(case.fact_texts.values()), self.k)
        return [key_by_text[t] for t in texts]


def tfidf_retrieval(cases: list[DecisionCase], k: int, expand: bool = False) -> RetrievedContext:
    corpus = [text for case in cases for text in case.fact_texts.values()]
    return RetrievedContext(TfidfRetriever(corpus), k, RETRIEVAL_QUERY_HINTS if expand else "")


FILTER_QUESTION = {
    "type": "noul",
    "instructions": "Does this observation indicate that something is wrong with the production service?",
    "criteria": {
        "true": "It reports an abnormal or degraded production metric, errors, a bad release, "
                "a recent change or user impact.",
        "false": "It is normal, routine, unrelated to production, or about planning, people or "
                 "other environments.",
    },
}


class ModelFilterContext:
    """Asks a classifier model, fact by fact, whether the fact signals a problem; keeps those
    at or above `threshold`. A learned alternative to retrieval, still outside the decision model."""

    def __init__(self, client, name: str, threshold: float):
        self.client = client  # anything with ask(state, questions) -> response, e.g. JevClient
        self.threshold = threshold
        self.name = f"{name}_filter_t{threshold}"

    def score(self, text: str) -> float:
        return self.client.ask(text, {"f": FILTER_QUESTION})["answers"]["f"]["noul"]

    def select(self, case: DecisionCase) -> list[str]:
        return [k for k, text in case.fact_texts.items() if self.score(text) >= self.threshold]
