import json

import httpx

from context_decisions.context.builder import build_context
from context_decisions.context.strategies import ModelFilterContext
from context_decisions.domain import POLICY_DESCRIPTIONS
from context_decisions.models.simple_jev import JevClient, SimpleJevModel
from context_decisions.schemas import DECISIONS


def fake_server(seen: list, fail_first: bool = True):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        if fail_first and len(seen) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        probs = {d: 0.05 for d in DECISIONS} | {"inspect_network": 0.7}
        return httpx.Response(200, json={"answers": {"decision": {
            "type": "choice", "choice": "inspect_network", "confidence": 0.3,
            "probabilities": probs}}, "usage": {"input_tokens": 900}})
    return httpx.MockTransport(handler)


def make_model(seen, **kwargs):
    model = SimpleJevModel("http://jev/v1", "org/small-model", **kwargs)
    model.jev.client = httpx.Client(transport=fake_server(seen))
    model.jev._backoff = lambda attempt: 0
    return model


def test_prompt_format_retry_and_result(cases):
    seen = []
    result = make_model(seen).decide(build_context(cases[0], list(cases[0].facts)))
    assert len(seen) == 2  # one 429, one success
    assert result.decision == "inspect_network"
    assert result.confidence == 0.7  # top probability, not the backend's own confidence
    body = seen[1]
    assert body["state"].startswith("TASK:")
    assert body["questions"]["decision"]["criteria"] == {d: None for d in DECISIONS}


def test_plain_format_with_policy_options(cases):
    seen = []
    case = cases[0]
    make_model(seen, state_format="plain", options="policy").decide(build_context(case, list(case.facts)))
    body = seen[1]
    assert body["state"].startswith(f"Incident report: {case.request}")
    assert "OPTIONS" not in body["state"] and "TASK" not in body["state"]
    assert all(t in body["state"] for t in case.fact_texts.values())
    assert body["questions"]["decision"]["criteria"] == POLICY_DESCRIPTIONS


class FakeFilterClient:
    def __init__(self, flagged: set[str]):
        self.flagged = flagged

    def ask(self, state, questions):
        return {"answers": {"f": {"noul": 0.9 if state in self.flagged else 0.1}}}


def test_model_filter_keeps_flagged_facts(cases):
    case = cases[0]
    keep = case.relevant_facts
    strategy = ModelFilterContext(FakeFilterClient({case.fact_texts[k] for k in keep}), "fake", 0.6)
    assert strategy.name == "fake_filter_t0.6"
    assert set(strategy.select(case)) == set(keep)
