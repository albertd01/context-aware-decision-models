"""Backend-independent metrics over run records."""
from __future__ import annotations

from statistics import mean, median

from context_decisions.schemas import ASK, DECISIONS, RunRecord


def per_class(records: list[RunRecord]) -> dict[str, dict[str, float]]:
    out = {}
    for d in DECISIONS:
        tp = sum(r.decision == d and r.correct_decision == d for r in records)
        predicted = sum(r.decision == d for r in records)
        actual = sum(r.correct_decision == d for r in records)
        p = tp / predicted if predicted else 0.0
        rc = tp / actual if actual else 0.0
        out[d] = {"precision": p, "recall": rc, "f1": 2 * p * rc / (p + rc) if p + rc else 0.0,
                  "support": actual}
    return out


def brier(records: list[RunRecord]) -> float | None:
    """Multiclass Brier score; needs a probability for every decision."""
    scored = [r for r in records if r.scores]
    if not scored:
        return None
    return mean(sum((r.scores.get(d, 0.0) - (d == r.correct_decision)) ** 2 for d in DECISIONS)
                for r in scored)


def ece(records: list[RunRecord], bins: int = 10) -> float | None:
    """Expected calibration error of the reported confidence."""
    rated = [r for r in records if r.confidence is not None]
    if not rated:
        return None
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        in_bin = [r for r in rated if lo < r.confidence <= hi or (b == 0 and r.confidence == 0)]
        if in_bin:
            acc = mean(r.decision == r.correct_decision for r in in_bin)
            conf = mean(r.confidence for r in in_bin)
            total += len(in_bin) / len(rated) * abs(acc - conf)
    return total


def relevant_recall(records: list[RunRecord]) -> float | None:
    """Share of relevant facts that made it into the context (cases with relevant facts only)."""
    with_relevant = [r for r in records if r.relevant_facts]
    if not with_relevant:
        return None
    return mean(len(set(r.selected_facts) & set(r.relevant_facts)) / len(r.relevant_facts)
                for r in with_relevant)


def summarize(records: list[RunRecord]) -> dict:
    classes = per_class(records)
    answerable = [r for r in records if r.correct_decision != ASK]
    return {
        "n": len(records),
        "accuracy": mean(r.decision == r.correct_decision for r in records),
        "macro_f1": mean(c["f1"] for c in classes.values()),
        # guessing vs abstaining: how often the model asked although the full facts decide
        "ask_rate_on_answerable": mean(r.decision == ASK for r in answerable) if answerable else None,
        "latency_ms_mean": mean(r.latency_ms for r in records),
        "latency_ms_median": median(r.latency_ms for r in records),
        "brier": brier(records),
        "ece": ece(records),
        "relevant_recall": relevant_recall(records),
        "per_class": classes,
    }
