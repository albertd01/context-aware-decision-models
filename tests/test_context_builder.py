from context_decisions.context.builder import build_context
from context_decisions.context.strategies import (FullContext, NoContext, OracleContext,
                                                  tfidf_retrieval)


def test_no_context_shows_no_facts(cases):
    ctx = build_context(cases[0], NoContext().select(cases[0]))
    assert ctx.facts == {}
    assert "- (none)" in ctx.context_text
    assert not any(t in ctx.context_text for t in cases[0].fact_texts.values())


def test_full_context_shows_every_fact(cases):
    case = cases[0]
    ctx = build_context(case, FullContext().select(case))
    assert ctx.facts == case.facts
    assert all(t in ctx.context_text for t in case.fact_texts.values())


def test_oracle_shows_only_relevant_facts(cases):
    for case in cases:
        assert set(OracleContext().select(case)) == set(case.relevant_facts)


def test_ground_truth_never_in_context(cases):
    for case in cases[:20]:
        ctx = build_context(case, FullContext().select(case))
        dumped = ctx.model_dump_json()
        assert "correct_decision" not in dumped and "relevant_facts" not in dumped
        # the label appears once, as one of the listed options, nowhere else
        assert ctx.context_text.count(case.correct_decision) == 1


def test_retrieval_returns_k_facts_of_the_case(cases):
    strategy = tfidf_retrieval(cases, k=3)
    for case in cases:
        selected = strategy.select(case)
        assert len(selected) == 3
        assert set(selected) <= set(case.facts)


def test_prompt_format(cases):
    text = build_context(cases[0], []).context_text
    for header in ("TASK:", "REQUEST:", "KNOWN FACTS:", "OPTIONS:", "Return exactly one decision."):
        assert header in text


def test_query_expansion_improves_relevant_recall(cases):
    def recall(strategy):
        hits = [len(set(strategy.select(c)) & set(c.relevant_facts)) / len(c.relevant_facts)
                for c in cases if c.relevant_facts]
        return sum(hits) / len(hits)

    plain, expanded = tfidf_retrieval(cases, k=3), tfidf_retrieval(cases, k=3, expand=True)
    assert expanded.name == "retrieval_expanded_k3"
    assert recall(expanded) > recall(plain) + 0.2
