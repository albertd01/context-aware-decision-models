"""Context strategies decide *which* facts a model sees. They return fact keys only;
turning them into a prompt is the builder's job."""
from __future__ import annotations

from context_decisions.context.retriever import TfidfRetriever
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
    def __init__(self, retriever: TfidfRetriever, k: int):
        self.retriever = retriever
        self.k = k
        self.name = f"retrieval_k{k}"

    def select(self, case: DecisionCase) -> list[str]:
        key_by_text = {text: key for key, text in case.fact_texts.items()}
        texts = self.retriever.retrieve(case.request, list(case.fact_texts.values()), self.k)
        return [key_by_text[t] for t in texts]


def tfidf_retrieval(cases: list[DecisionCase], k: int) -> RetrievedContext:
    corpus = [text for case in cases for text in case.fact_texts.values()]
    return RetrievedContext(TfidfRetriever(corpus), k)
