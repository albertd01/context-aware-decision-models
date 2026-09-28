import pytest

from context_decisions.context.strategies import FullContext, NoContext, OracleContext
from context_decisions.evaluation.metrics import brier, ece, per_class, summarize
from context_decisions.evaluation.runner import run
from context_decisions.models.rule_baseline import RuleDecisionModel
from context_decisions.schemas import ASK, RunRecord


def rec(decision, correct, confidence=None, scores=None, selected=(), relevant=()):
    return RunRecord(case_id="c", experiment="e", model="m", decision=decision,
                     correct_decision=correct, confidence=confidence, scores=scores,
                     latency_ms=1.0, selected_facts=list(selected), relevant_facts=list(relevant))


def test_per_class_precision_recall():
    records = [rec("escalate", "escalate"), rec("escalate", "inspect_network"),
               rec("inspect_network", "inspect_network")]
    classes = per_class(records)
    assert classes["escalate"]["precision"] == 0.5
    assert classes["escalate"]["recall"] == 1.0
    assert classes["inspect_network"]["recall"] == 0.5


def test_brier_perfect_and_worst():
    assert brier([rec("escalate", "escalate", scores={"escalate": 1.0})]) == 0.0
    assert brier([rec("inspect_network", "escalate", scores={"inspect_network": 1.0})]) == 2.0
    assert brier([rec("escalate", "escalate")]) is None


def test_ece():
    assert ece([rec("escalate", "escalate", confidence=1.0)]) == 0.0
    assert ece([rec("escalate", "inspect_network", confidence=0.9)]) == pytest.approx(0.9)
    assert ece([rec("escalate", "escalate")]) is None


def test_summary_recall_and_ask_rate():
    records = [rec(ASK, "escalate", selected=["a"], relevant=["a", "b"]),
               rec("escalate", "escalate", selected=["a", "b"], relevant=["a", "b"])]
    s = summarize(records)
    assert s["accuracy"] == 0.5
    assert s["ask_rate_on_answerable"] == 0.5
    assert s["relevant_recall"] == 0.75


def test_rule_baseline_is_perfect_with_full_and_oracle_context(cases):
    for strategy in (FullContext(), OracleContext()):
        assert summarize(run(cases, strategy, RuleDecisionModel()))["accuracy"] == 1.0


def test_rule_baseline_asks_without_context(cases):
    records = run(cases, NoContext(), RuleDecisionModel())
    assert all(r.decision == ASK for r in records)


def test_runner_writes_jsonl(cases, tmp_path):
    run(cases[:5], NoContext(), RuleDecisionModel(), results_dir=tmp_path)
    lines = (tmp_path / "no_context__rules.jsonl").read_text().splitlines()
    assert len(lines) == 5
