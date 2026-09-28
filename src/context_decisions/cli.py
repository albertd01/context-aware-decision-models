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

from context_decisions.context.strategies import (FullContext, ModelFilterContext, NoContext,
                                                  OracleContext, tfidf_retrieval)
from context_decisions.dataset import load_cases, load_config, save_jsonl
from context_decisions.evaluation.metrics import summarize
from context_decisions.evaluation.runner import run
from context_decisions.generator import generate, request_leakage
from context_decisions.models.rule_baseline import RuleDecisionModel
from context_decisions.models.simple_jev import JevClient, SimpleJevModel
from context_decisions.schemas import RunRecord

EXPERIMENTS = ["no-context", "full-context", "oracle", "retrieval", "retrieval-expanded", "model-filter"]


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
        case "model-filter":
            f = cfg["filter"]
            preset = cfg["models"][f["model"]]
            client = JevClient(preset["url"], preset["model"],
                               min_interval_s=preset.get("min_interval_s", 0.0))
            return ModelFilterContext(client, f["model"], f["threshold"])
    raise SystemExit(f"unknown experiment {name!r}; choose from {EXPERIMENTS}")


def make_model(name: str, cfg: dict):
    preset = cfg["models"].get(name)
    if preset is None:
        raise SystemExit(f"unknown model {name!r}; choose from {list(cfg['models'])}")
    if preset["backend"] == "rules":
        return RuleDecisionModel()
    if preset["backend"] == "simple_jev":
        return SimpleJevModel(preset["url"], preset["model"], name=name,
                              min_interval_s=preset.get("min_interval_s", 0.0),
                              state_format=preset.get("state_format", "prompt"),
                              options=preset.get("options", "bare"))
    raise SystemExit(f"unknown backend {preset['backend']!r}")


def cmd_generate(cfg: dict, args) -> None:
    path = args.out or cfg["dataset"]["path"]
    cases = generate(cfg["dataset"], cfg["seed"] if args.seed is None else args.seed)
    save_jsonl(cases, path)
    print(f"wrote {len(cases)} cases to {path}")
    print("classes:", dict(sorted(Counter(c.correct_decision for c in cases).items())))
    print(f"request-only leakage accuracy: {request_leakage(cases):.2f} (chance 0.14)")


def cmd_run(cfg: dict, args) -> None:
    cases = load_cases(args.cases or cfg["dataset"]["path"])
    if args.limit:
        cases = cases[:args.limit]
    model = make_model(args.model, cfg)
    for name in EXPERIMENTS if args.experiment == "all" else [args.experiment]:
        records = run(cases, make_strategy(name, cases, cfg), model,
                      results_dir=args.results_dir or cfg["evaluation"]["results_dir"],
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
    if "_filter_" in name:
        return 3
    order = ["no_context", "retrieval_k", "retrieval_expanded_k", "", "oracle_context", "full_context"]
    return next((i for i, prefix in enumerate(order) if prefix and name.startswith(prefix)), len(order))


def cmd_compare(cfg: dict, args) -> None:
    results_dir = Path(cfg["evaluation"]["results_dir"])
    table = comparison_table(results_dir)
    (results_dir / "comparison.md").write_text(table + "\n")
    print(table)


def main() -> None:
    parser = argparse.ArgumentParser(prog="context_decisions")
    parser.add_argument("--config", default="config/default.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate-data")
    g.add_argument("--seed", type=int, help="override the config seed (e.g. for a dev set)")
    g.add_argument("--out", help="output path instead of dataset.path")
    g.set_defaults(fn=cmd_generate)
    p = sub.add_parser("run")
    p.add_argument("--experiment", required=True, choices=[*EXPERIMENTS, "all"])
    p.add_argument("--model", default="rules")
    p.add_argument("--limit", type=int, help="only the first N cases (smoke tests)")
    p.add_argument("--cases", help="dataset path instead of dataset.path (e.g. a dev set)")
    p.add_argument("--results-dir", help="results directory instead of evaluation.results_dir")
    p.set_defaults(fn=cmd_run)
    sub.add_parser("compare").set_defaults(fn=cmd_compare)
    args = parser.parse_args()
    args.fn(load_config(args.config), args)


if __name__ == "__main__":
    main()
