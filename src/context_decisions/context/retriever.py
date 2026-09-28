"""Minimal lexical retriever. No vector database."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    def __init__(self, corpus: list[str]):
        """`corpus` fixes the vocabulary and IDF weights, e.g. every fact sentence in the dataset."""
        self.vectorizer = TfidfVectorizer(stop_words="english").fit(corpus)

    def retrieve(self, request: str, facts: list[str], k: int) -> list[str]:
        if not facts or k <= 0:
            return []
        sims = cosine_similarity(self.vectorizer.transform([request]),
                                 self.vectorizer.transform(facts))[0]
        order = sorted(range(len(facts)), key=lambda i: -sims[i])  # stable: ties keep fact order
        return [facts[i] for i in order[:k]]
