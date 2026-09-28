"""Deterministic synthetic case generation. No LLM involved.

Each case is built *for* a target decision, then checked against the rule set, so data and
rules cannot drift apart.
"""
from __future__ import annotations

import random
from collections import Counter

from context_decisions import domain
from context_decisions.domain import APPLICATION_ERRORS, BENIGN, Facts
from context_decisions.schemas import ASK, DECISIONS, DecisionCase


def _signal(target: str, rng: random.Random, hard: bool) -> Facts:
    """Diagnostic facts for one case: benign baseline plus the target's signal."""
    f: Facts = dict(BENIGN)
    f["customer_impact"] = rng.choice(["low", "high"])

    if target == "escalate":
        f["customer_impact"] = "high"
        for key in rng.sample(["database_latency", "packet_loss", "infra"], k=2):
            if key == "infra":
                f[rng.choice(["cpu_usage", "memory_usage"])] = "saturated"
            else:
                f[key] = "high"
    elif target == "rollback_deployment":
        f.update(deployment_recent=True, known_bad_deployment=True, regression_severity="severe")
        if hard:  # the bad release also throws errors: application rule is tempting
            f["application_errors"] = rng.choice(APPLICATION_ERRORS)
    elif target == "inspect_application":
        f.update(deployment_recent=True, known_bad_deployment=False,
                 application_errors=rng.choice(APPLICATION_ERRORS),
                 regression_severity=rng.choice(["none", "moderate"]))
        if hard:  # flagged release, but regression not severe: rollback is tempting
            f["known_bad_deployment"] = True
            f["regression_severity"] = "moderate"
    elif target == "inspect_database":
        f["database_latency"] = "high"
        if hard:  # fresh deployment without errors: application is tempting
            f.update(deployment_recent=True, known_bad_deployment=False)
    elif target == "inspect_network":
        f["packet_loss"] = "high"
        if hard:
            f.update(deployment_recent=True, known_bad_deployment=False,
                     regression_severity="moderate")
    elif target == "inspect_infrastructure":
        for key in rng.sample(["cpu_usage", "memory_usage"], k=rng.choice([1, 2])):
            f[key] = "saturated"
        if hard:  # slow database caused by saturated app servers: database is tempting
            f["cpu_usage"] = "saturated"
            f["database_latency"] = "high"
            f["customer_impact"] = "low"
    elif target == ASK:
        return _incomplete_signal(rng)
    else:
        raise ValueError(target)
    return f


def _incomplete_signal(rng: random.Random) -> Facts:
    """A symptom whose deciding fact is unknown, plus a few benign facts."""
    pattern = rng.choice(["db_without_cpu", "deploy_without_logs", "regression_without_deploy",
                          "impact_only"])
    if pattern == "db_without_cpu":
        f: Facts = {"database_latency": "high"}
        benign_pool = ["memory_usage", "packet_loss", "application_errors"]
    elif pattern == "deploy_without_logs":
        f = {"deployment_recent": True, "regression_severity": rng.choice(["moderate", "severe"])}
        benign_pool = ["database_latency", "cpu_usage", "packet_loss"]
    elif pattern == "regression_without_deploy":
        f = {"regression_severity": "severe", "application_errors": rng.choice(APPLICATION_ERRORS)}
        benign_pool = ["database_latency", "cpu_usage", "memory_usage", "packet_loss"]
    else:
        f = {"customer_impact": "high", "regression_severity": "moderate"}
        benign_pool = ["database_latency", "cpu_usage"]
    for key in rng.sample(benign_pool, k=rng.randint(0, 2)):
        f[key] = BENIGN[key]
    return f


def make_case(index: int, target: str, rng: random.Random, cfg: dict) -> DecisionCase:
    hard = rng.random() < cfg["hard_case_prob"]
    facts = _signal(target, rng, hard)
    decision, relevant = domain.decide_with_reason(facts)

    if target != ASK:  # leave some non-decisive diagnostic facts unknown
        for key in [k for k in facts if k not in relevant]:
            if rng.random() < cfg["hide_background_prob"]:
                del facts[key]
    if facts.get("deployment_recent") is not True:
        facts.pop("known_bad_deployment", None)

    decision, relevant = domain.decide_with_reason(facts)
    assert decision == target, f"case {index}: built for {target}, rules say {decision}"

    distractors = rng.sample(sorted(domain.DISTRACTORS), k=cfg["distractors_per_case"])
    keys = [*facts, *distractors]
    rng.shuffle(keys)
    all_facts = {k: facts.get(k, True) for k in keys}  # distractors are simply "true"

    return DecisionCase(
        id=f"case_{index:04d}",
        request=rng.choice(domain.REQUESTS),
        facts=all_facts,
        fact_texts={k: domain.render_fact(k, v, rng) for k, v in all_facts.items()},
        relevant_facts=relevant,
        irrelevant_facts=distractors,
        correct_decision=decision,
        requires_clarification=decision == ASK,
    )


def generate(cfg: dict, seed: int) -> list[DecisionCase]:
    """Round-robin over decisions so classes are balanced to within one case."""
    rng = random.Random(seed)
    targets = [DECISIONS[i % len(DECISIONS)] for i in range(cfg["num_cases"])]
    rng.shuffle(targets)
    return [make_case(i, t, rng, cfg) for i, t in enumerate(targets)]


def request_leakage(cases: list[DecisionCase], folds: int = 5) -> float:
    """Held-out accuracy of predicting the decision from the request text alone.

    Near chance (1/7) means the request does not give the answer away.
    """
    correct = 0
    for fold in range(folds):
        train = [c for i, c in enumerate(cases) if i % folds != fold]
        test = [c for i, c in enumerate(cases) if i % folds == fold]
        overall = Counter(c.correct_decision for c in train).most_common(1)[0][0]
        by_request: dict[str, Counter] = {}
        for c in train:
            by_request.setdefault(c.request, Counter())[c.correct_decision] += 1
        for c in test:
            counts = by_request.get(c.request)
            guess = counts.most_common(1)[0][0] if counts else overall
            correct += guess == c.correct_decision
    return correct / len(cases)
