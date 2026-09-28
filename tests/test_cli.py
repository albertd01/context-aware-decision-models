from context_decisions.cli import comparison_table
from context_decisions.context.strategies import FullContext, NoContext
from context_decisions.evaluation.runner import run
from context_decisions.models.rule_baseline import RuleDecisionModel


def test_comparison_table_lists_every_run_in_condition_order(cases, tmp_path):
    for strategy in (FullContext(), NoContext()):
        run(cases[:20], strategy, RuleDecisionModel(), results_dir=tmp_path)
    table = comparison_table(tmp_path)
    rows = [line for line in table.splitlines() if line.startswith("| rules")]
    assert [r.split("|")[2].strip() for r in rows] == ["no_context", "full_context"]
    assert "| 1.00 |" in rows[1]
