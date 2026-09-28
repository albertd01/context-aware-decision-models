"""simple-jev backend: the model scores each decision label's logit; nothing is generated.

Works against a local `simple-jev` server (e.g. `--backend laya`) or the public demo API
(https://simple-jev-demo-api.featherless.ai/v1, 2 requests/s, 2k-token context).
"""
from __future__ import annotations

import time

import httpx

from context_decisions.domain import ACTION_DESCRIPTIONS, POLICY_DESCRIPTIONS
from context_decisions.schemas import DecisionContext, DecisionResult

PROMPT_INSTRUCTIONS = "Which next action is most appropriate?"
PLAIN_INSTRUCTIONS = ("Given the incident report and the known facts, "
                      "which triage action should be taken next?")
OPTION_DESCRIPTIONS = {"bare": {}, "actions": ACTION_DESCRIPTIONS, "policy": POLICY_DESCRIPTIONS}


class JevClient:
    """POSTs classifier requests with a client-side rate limit and retries on busy servers."""

    def __init__(self, url: str, model: str, api_key: str | None = None,
                 min_interval_s: float = 0.0, retries: int = 10):
        self.url = url.rstrip("/") + "/classifier"
        self.model = model
        self.min_interval_s = min_interval_s
        self.retries = retries
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(headers=headers, timeout=120)
        self._last_call = 0.0

    def ask(self, state: str, questions: dict) -> dict:
        return self._post({"model": self.model, "state": state, "questions": questions})

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
                time.sleep(self._backoff(attempt))
                continue
            if r.status_code in (429, 502, 503, 504) and attempt < self.retries:
                time.sleep(max(float(r.headers.get("Retry-After", 0)), self._backoff(attempt)))
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError("unreachable")

    @staticmethod
    def _backoff(attempt: int) -> float:
        """1, 2, 4, ... seconds, capped at 60: busy demo queues can take minutes to drain."""
        return min(2 ** attempt, 60)


def plain_state(context: DecisionContext) -> str:
    """Request and facts only: no task text, no option list. Suits encoder models like Laya."""
    facts = "\n".join(f"- {t}" for t in context.fact_texts) or "- (none)"
    return f"Incident report: {context.request}\nKnown facts:\n{facts}"


class SimpleJevModel:
    """`state_format`: "prompt" sends the standard prompt, "plain" only request + facts.
    `options`: "bare" labels, "actions" (what each does) or "policy" (when to choose it)."""

    def __init__(self, url: str, model: str, name: str | None = None, api_key: str | None = None,
                 min_interval_s: float = 0.0, retries: int = 10,
                 state_format: str = "prompt", options: str = "bare"):
        self.jev = JevClient(url, model, api_key, min_interval_s, retries)
        self.name = name or model.split("/")[-1]
        self.state_format = state_format
        self.descriptions = OPTION_DESCRIPTIONS[options]

    def decide(self, context: DecisionContext) -> DecisionResult:
        plain = self.state_format == "plain"
        response = self.jev.ask(
            plain_state(context) if plain else context.context_text,
            {"decision": {
                "type": "choice",
                "instructions": PLAIN_INSTRUCTIONS if plain else PROMPT_INSTRUCTIONS,
                "criteria": {d: self.descriptions.get(d) for d in context.available_decisions},
            }},
        )
        answer = response["answers"]["decision"]
        probs = answer["probabilities"]
        return DecisionResult(
            decision=answer["choice"],
            # backends disagree on "confidence" (Laya: 1 - normalized entropy); use the top probability
            confidence=max(probs.values()),
            scores=probs,
            raw_output={"usage": response.get("usage"), "backend_confidence": answer.get("confidence")},
        )
