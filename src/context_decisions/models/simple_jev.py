"""simple-jev backend: the model scores each decision label's logit; nothing is generated.

Works against a local `simple-jev` server or the public demo API
(https://simple-jev-demo-api.featherless.ai/v1, 2 requests/s, 2k-token context).
"""
from __future__ import annotations

import time

import httpx

from context_decisions.schemas import DecisionContext, DecisionResult

INSTRUCTIONS = "Which next action is most appropriate?"


class SimpleJevModel:
    def __init__(self, url: str, model: str, name: str | None = None, api_key: str | None = None,
                 min_interval_s: float = 0.0, retries: int = 5):
        self.url = url.rstrip("/") + "/classifier"
        self.model = model
        self.name = name or model.split("/")[-1]
        self.min_interval_s = min_interval_s  # client-side rate limit
        self.retries = retries
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(headers=headers, timeout=120)
        self._last_call = 0.0

    def decide(self, context: DecisionContext) -> DecisionResult:
        body = {
            "model": self.model,
            "state": context.context_text,
            "questions": {"decision": {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": {d: None for d in context.available_decisions},
            }},
        }
        response = self._post(body)
        answer = response["answers"]["decision"]
        return DecisionResult(
            decision=answer["choice"],
            confidence=answer["confidence"],
            scores=answer["probabilities"],
            raw_output={"usage": response.get("usage")},
        )

    def _post(self, body: dict) -> dict:
        for attempt in range(self.retries + 1):
            wait = self._last_call + self.min_interval_s - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                r = self.client.post(self.url, json=body)
            except httpx.TransportError:
                if attempt == self.retries:
                    raise
                time.sleep(2 ** attempt)
                continue
            if r.status_code in (429, 502, 503, 504) and attempt < self.retries:
                time.sleep(float(r.headers.get("Retry-After", 2 ** attempt)))
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError("unreachable")
