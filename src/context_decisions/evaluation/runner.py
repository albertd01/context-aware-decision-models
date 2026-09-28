"""Runs one (context strategy, model) pair over a dataset and stores every decision."""
from __future__ import annotations

import time
from pathlib import Path

from context_decisions.context.builder import build_context
from context_decisions.dataset import save_jsonl
from context_decisions.models.base import DecisionModel
from context_decisions.schemas import DecisionCase, RunRecord


def run(cases: list[DecisionCase], strategy, model: DecisionModel,
        results_dir: str | Path | None = None, save_raw_outputs: bool = True) -> list[RunRecord]:
    records = []
    for case in cases:
        selected = strategy.select(case)
        context = build_context(case, selected)  # ground truth never enters the context
        start = time.perf_counter()
        result = model.decide(context)
        elapsed_ms = (time.perf_counter() - start) * 1000
        records.append(RunRecord(
            case_id=case.id,
            experiment=strategy.name,
            model=model.name,
            decision=result.decision,
            correct_decision=case.correct_decision,
            confidence=result.confidence,
            scores=result.scores,
            latency_ms=result.latency_ms if result.latency_ms is not None else elapsed_ms,
            selected_facts=selected,
            relevant_facts=case.relevant_facts,
            raw_output=result.raw_output if save_raw_outputs else None,
        ))
    if results_dir is not None:
        save_jsonl(records, Path(results_dir) / f"{strategy.name}__{model.name}.jsonl")
    return records
