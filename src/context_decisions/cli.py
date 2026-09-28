"""Command line entry point.

    python -m context_decisions.cli generate-data
    python -m context_decisions.cli run --experiment full-context --model qwen-4b
    python -m context_decisions.cli run --experiment all --model rules
    python -m context_decisions.cli compare
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from context_decisions.context.strategies import (FullContext, NoContext, OracleContext,
                                                  tfidf_retrieval)
from context_decisions.dataset import load_cases, load_config, save_jsonl
from context_decisions.evaluation.metrics import summarize
from context_decisions.evaluation.runner import run
from context_decisions.generator import generate, request_leakage
from context_decisions.models.rule_baseline import RuleDecisionModel
from context_decisions.models.simple_jev import SimpleJevModel
from context_decisions.schemas import RunRecord

EXPERIMENTS = ["no-context", "full-context", "oracle", "retrieval", "retrieval-expanded"]


def make_strategy(name: str, cases, cfg: dict):
    k = cfg["retrieval"]["k"]
    match name:
        case "no-context":
            return NoContext()
        case "full-context":
            return FullContext()
        case "oracle":
            return OracleContext()
        case "retrieval":
            return tfidf_retrieval(cases, k)
        case "retrieval-expanded":
            return tfidf_retrieval(cases, k, expand=True)
    raise SystemExit(f"unknown experiment {name!r}; choose from {EXPERIMENTS}")


def make_model(name: str, cfg: dict):
    preset = cfg["models"].get(name)
    if preset is None:
        raise SystemExit(f"unknown model {name!r}; choose from {list(cfg['models'])}")
    if preset["backend"] == "rules":
        return RuleDecisionModel()
    if preset["backend"] == "simple_jev":
        return SimpleJevModel(preset["url"], preset["model"], name=name,
                              min_interval_s=preset.get("min_interval_s", 0.0))
    raise SystemExit(f"unknown backend {preset['backend']!r}")


def cmd_generate(cfg: dict, args) -> None:
    cases = generate(cfg["dataset"], cfg["seed"])
    save_jsonl(cases, cfg["dataset"]["path"])
    print(f"wrote {len(cases)} cases to {cfg['dataset']['path']}")
    print("classes:", dict(sorted(Counter(c.correct_decision for c in cases).items())))
    print(f"request-only leakage accuracy: {request_leakage(cases):.2f} (chance 0.14)")


def cmd_run(cfg: dict, args) -> None:
    cases = load_cases(cfg["dataset"]["path"])
    if args.limit:
        cases = cases[:args.limit]
    model = make_model(args.model, cfg)
    for name in EXPERIMENTS if args.experiment == "all" else [args.experiment]:
        records = run(cases, make_strategy(name, cases, cfg), model,
                      results_dir=cfg["evaluation"]["results_dir"],
                      save_raw_outputs=cfg["evaluation"]["save_raw_outputs"])
        s = summarize(records)
        print(f"{name:20} {model.name:10} acc={s['accuracy']:.2f} macroF1={s['macro_f1']:.2f} "
              f"latency={s['latency_ms_mean']:.0f}ms")


def comparison_table(results_dir: Path) -> str:
    rows = []
    for path in sorted(results_dir.glob("*__*.jsonl")):
        records = [RunRecord.model_validate_json(l) for l in path.read_text().splitlines() if l]
        if records:
            rows.append((records[0].model, records[0].experiment, summarize(records)))
    fmt = lambda v, spec=".2f": "–" if v is None else format(v, spec)  # noqa: E731
    lines = ["| Model | Condition | N | Accuracy | Macro F1 | Brier | ECE | Ask rate* | Relevant recall | Latency (ms) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for model, experiment, s in sorted(rows, key=lambda r: (r[0], experiment_order(r[1]))):
        lines.append(f"| {model} | {experiment} | {s['n']} | {fmt(s['accuracy'])} | {fmt(s['macro_f1'])} "
                     f"| {fmt(s['brier'])} | {fmt(s['ece'])} | {fmt(s['ask_rate_on_answerable'])} "
                     f"| {fmt(s['relevant_recall'])} | {fmt(s['latency_ms_median'], '.0f')} |")
    lines.append("\n*Ask rate: share of answerable cases where the model chose ask_for_more_information.")
    return "\n".join(lines)


def experiment_order(name: str) -> int:
    order = ["no_context", "retrieval_k", "retrieval_expanded_k", "oracle_context", "full_context"]
    return next((i for i, prefix in enumerate(order) if name.startswith(prefix)), len(order))


def cmd_compare(cfg: dict, args) -> None:
    results_dir = Path(cfg["evaluation"]["results_dir"])
    table = comparison_table(results_dir)
    (results_dir / "comparison.md").write_text(table + "\n")
    print(table)


def main() -> None:
    parser = argparse.ArgumentParser(prog="context_decisions")
    parser.add_argument("--config", default="config/default.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("generate-data").set_defaults(fn=cmd_generate)
    p = sub.add_parser("run")
    p.add_argument("--experiment", required=True, choices=[*EXPERIMENTS, "all"])
    p.add_argument("--model", default="rules")
    p.add_argument("--limit", type=int, help="only the first N cases (smoke tests)")
    p.set_defaults(fn=cmd_run)
    sub.add_parser("compare").set_defaults(fn=cmd_compare)
    args = parser.parse_args()
    args.fn(load_config(args.config), args)


if __name__ == "__main__":
    main()
