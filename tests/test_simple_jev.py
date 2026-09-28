import httpx

from context_decisions.context.builder import build_context
from context_decisions.models.simple_jev import SimpleJevModel
from context_decisions.schemas import DECISIONS


def fake_server(seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        body = httpx.Request("POST", "x", content=request.content).read()
        seen.append(body)
        if len(seen) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        probs = {d: 0.0 for d in DECISIONS} | {"inspect_network": 1.0}
        return httpx.Response(200, json={"answers": {"decision": {
            "type": "choice", "choice": "inspect_network", "confidence": 1.0,
            "probabilities": probs}}, "usage": {"input_tokens": 900}})
    return httpx.MockTransport(handler)


def test_request_shape_retry_and_result(cases):
    seen = []
    model = SimpleJevModel("http://jev/v1", "org/small-model")
    model.client = httpx.Client(transport=fake_server(seen))
    model._backoff = lambda attempt: 0
    result = model.decide(build_context(cases[0], list(cases[0].facts)))

    assert len(seen) == 2  # one 429, one success
    assert model.name == "small-model"
    assert result.decision == "inspect_network"
    assert result.scores["inspect_network"] == 1.0
    assert b'"criteria"' in seen[1] and cases[0].correct_decision.encode() in seen[1]
