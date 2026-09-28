import random
from collections import Counter

from context_decisions import domain
from context_decisions.dataset import load_cases, save_jsonl
from context_decisions.generator import generate, request_leakage
from context_decisions.schemas import ASK, DECISIONS
from conftest import CFG


def test_same_seed_same_dataset(cases):
    assert generate(CFG, seed=42) == cases
    assert generate(CFG, seed=7) != cases


def test_classes_balanced(cases):
    counts = Counter(c.correct_decision for c in cases)
    assert set(counts) == set(DECISIONS)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_labels_match_rules(cases):
    for c in cases:
        assert domain.decide(c.facts) == c.correct_decision
        assert c.requires_clarification == (c.correct_decision == ASK)


def test_relevant_facts_alone_are_sufficient(cases):
    for c in cases:
        assert domain.decide({k: c.facts[k] for k in c.relevant_facts}) == c.correct_decision


PRECEDENCE = ["escalate", "rollback_deployment", "inspect_application", "inspect_database",
              "inspect_network", "inspect_infrastructure", ASK]


def test_rules_are_monotone(cases):
    """Dropping facts never makes a rule fire, it can only fall through to a later rule."""
    rng = random.Random(0)
    for c in cases:
        subset = {k: v for k, v in c.facts.items() if rng.random() < 0.5}
        weaker = domain.decide(subset)
        assert weaker == c.correct_decision or \
            PRECEDENCE.index(weaker) > PRECEDENCE.index(c.correct_decision), c.id


def test_distractors_do_not_change_decision(cases):
    for c in cases:
        without = {k: v for k, v in c.facts.items() if k not in c.irrelevant_facts}
        assert domain.decide(without) == c.correct_decision


def test_request_does_not_leak_label(cases):
    assert request_leakage(cases) < 0.25  # chance is 1/7


def test_every_fact_has_distinct_text(cases):
    for c in cases:
        assert set(c.fact_texts) == set(c.facts)
        assert len(set(c.fact_texts.values())) == len(c.fact_texts)


def test_jsonl_round_trip(cases, tmp_path):
    save_jsonl(cases, tmp_path / "cases.jsonl")
    assert load_cases(tmp_path / "cases.jsonl") == cases
