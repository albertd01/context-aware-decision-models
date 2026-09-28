# IMPLEMENTATION_PLAN.md

## 1. Goal

Build a small experimental framework to evaluate how small decision models use domain knowledge.

The project should compare:

1. no additional context,
2. full context injection,
3. retrieved context,
4. optionally a fine-tuned or adapted decision model,
5. a larger reasoning model as reference.

Main research question:

> **How should domain knowledge be provided to small decision models: through runtime context, retrieval, or model weights?**

The first version should be simple, reproducible, and easy to extend.

## 2. Initial Scope

Use a synthetic **DevOps incident triage** domain.

Each test case contains a user request, domain facts, relevant facts, optional irrelevant facts, a correct decision, and optionally a clarification requirement.

Example:

```json
{
  "request": "The API became slow after the latest deployment.",
  "facts": {
    "deployment_recent": true,
    "database_latency": "normal",
    "cpu_usage": "normal",
    "application_errors": "schema mismatch"
  },
  "relevant_facts": [
    "deployment_recent",
    "application_errors"
  ],
  "correct_decision": "inspect_application",
  "requires_clarification": false
}
```

Use controlled synthetic cases so the ground truth is known.

## 3. Initial Decision Space

```text
inspect_application
inspect_database
inspect_network
inspect_infrastructure
rollback_deployment
ask_for_more_information
escalate
```

Do not add more classes until the first experiment works.

## 4. Core Data Schemas

Use Pydantic or dataclasses.

### `DecisionCase`

```python
class DecisionCase:
    id: str
    request: str
    facts: dict[str, object]
    relevant_facts: list[str]
    irrelevant_facts: list[str]
    correct_decision: str
    requires_clarification: bool = False
```

### `DecisionContext`

```python
class DecisionContext:
    request: str
    context_text: str
    available_decisions: list[str]
```

### `DecisionResult`

```python
class DecisionResult:
    decision: str
    confidence: float | None = None
    scores: dict[str, float] | None = None
    raw_output: object | None = None
    latency_ms: float | None = None
```

## 5. Decision Model Interface

All models must implement the same interface.

```python
class DecisionModel(Protocol):
    def decide(
        self,
        context: DecisionContext
    ) -> DecisionResult:
        ...
```

The evaluation code must not depend directly on a specific model implementation.

Possible later models:

```text
rules
Needle
Jev-like model
small Qwen
large Qwen
fine-tuned model
```

## 6. Dataset Generation

Start with deterministic synthetic generation. Do not use an LLM for the first dataset version.

Example rules:

```text
recent deployment + application errors
→ inspect_application

database latency high + application CPU normal
→ inspect_database

packet loss high
→ inspect_network

CPU or memory saturation
→ inspect_infrastructure

recent deployment + severe regression + known bad deployment
→ rollback_deployment

insufficient facts
→ ask_for_more_information
```

Generate about 200 cases initially and aim for rough balance across classes.

## 7. Case Variants

For every base case, generate:

```text
base_case
├── no_context
├── relevant_context
├── full_context
├── irrelevant_context
└── conflicting_context
```

The same underlying decision should be tested under different context conditions.

## 8. No-Context Builder

```python
class NoContextBuilder:
    def build(self, case: DecisionCase) -> DecisionContext:
        ...
```

No domain facts are included.

## 9. Full-Context Builder

```python
class FullContextBuilder:
    def build(self, case: DecisionCase) -> DecisionContext:
        ...
```

Include every available fact.

## 10. Retrieval Condition

Implement a minimal retriever. Do not start with a vector database.

Recommended first version:

```text
TF-IDF
```

or:

```text
sentence-transformer cosine similarity
```

Interface:

```python
class Retriever:
    def retrieve(
        self,
        request: str,
        facts: list[str],
        k: int
    ) -> list[str]:
        ...
```

Start with `k = 3`.

## 11. Decision Context Builder

Keep retrieval separate from prompt construction.

```python
class DecisionContextBuilder:
    def build(
        self,
        case: DecisionCase,
        selected_facts: list[str]
    ) -> DecisionContext:
        ...
```

Standardized prompt format:

```text
TASK:
Select the most appropriate next action.

REQUEST:
The API became slow after the latest deployment.

KNOWN FACTS:
- deployment happened 20 minutes ago
- database latency is normal
- schema mismatch errors appear in application logs

OPTIONS:
- inspect_application
- inspect_database
- inspect_network
- inspect_infrastructure
- rollback_deployment
- ask_for_more_information
- escalate

Return exactly one decision.
```

## 12. Rule Baseline

Implement a deterministic baseline first.

```python
class RuleDecisionModel:
    ...
```

Use the same rule system that generated the synthetic dataset.

Expected accuracy should be close to 100%. If not, the data or rules are inconsistent.

## 13. Small Model Baseline

Implement a generic LLM decision model with structured output.

Example:

```json
{
  "decision": "inspect_database",
  "confidence": 0.82
}
```

Do not require chain-of-thought.

Possible initial backends:

```text
small Qwen
small local instruct model
Jev-like implementation
```

Keep the backend configurable.

## 14. Large Model Reference

Use a larger reasoning-capable model as reference.

Compare:

```text
accuracy
latency
context sensitivity
confidence behavior
```

## 15. First Three Experiments

### Experiment 1 — No Context

```text
request
→ small decision model
→ decision
```

Measure accuracy, latency, confidence.

### Experiment 2 — Full Context

```text
request + all facts
→ small decision model
→ decision
```

### Experiment 3 — Retrieved Context

```text
request
→ retriever
→ top-k facts
→ small decision model
→ decision
```

Compare all three conditions.

## 16. Evaluation Metrics

At minimum:

```text
accuracy
per-class precision
per-class recall
macro F1
latency
```

If confidence is available:

```text
Brier score
Expected Calibration Error
```

Also report accuracy by context condition.

## 17. Context Robustness Experiment

Add irrelevant facts to otherwise valid cases.

Test:

```text
N = 0
N = 2
N = 5
N = 10
```

Plot:

```text
accuracy vs irrelevant context size
```

## 18. Missing Information / Clarification

Add cases where the correct decision cannot be made reliably.

Example:

```text
Request:
"The API is slow."

Known:
CPU normal

Unknown:
database latency
deployment status
network health
```

Correct decision:

```text
ask_for_more_information
```

Measure whether the model guesses or abstains correctly.

## 19. Optional Information-Seeking Experiment

Later extend the decision space with:

```text
check_database_latency
check_deployment_history
check_network_health
check_application_logs
```

Then test:

```text
partial context
→ decision model
→ request next fact
→ update context
→ decision model
→ final decision
```

Do this only after the basic benchmark works.

## 20. Confidence-Based Escalation

Later implement:

```text
AUTO
REASON
ASK
```

Placeholder logic:

```python
if confidence >= 0.85:
    route = "AUTO"
elif confidence >= 0.55:
    route = "REASON"
else:
    route = "ASK"
```

Thresholds must be calibrated empirically.

## 21. Two-Stage Decision Pipeline

Test:

```text
Request
   ↓
Small Decision Model
   ↓
confidence
   ├── high → accept
   └── low → Large Model
```

Compare against always using the large model.

Measure:

```text
accuracy
average latency
large-model calls per case
```

## 22. Fine-Tuning Phase

Do not start here.

Only begin once the dataset and evaluation are stable.

Compare:

```text
base model + retrieval
fine-tuned model + retrieval
fine-tuned model + reduced context
```

Main question:

> **Can stable decision patterns move into the weights while dynamic facts remain in runtime context?**

Possible methods:

```text
LoRA
small classifier head
instruction fine-tuning
```

## 23. Result Storage

Store every run as JSONL.

Example:

```json
{
  "case_id": "case_0142",
  "experiment": "retrieval_k3",
  "model": "small_qwen",
  "decision": "inspect_database",
  "correct_decision": "inspect_database",
  "confidence": 0.87,
  "latency_ms": 74,
  "retrieved_facts": [
    "database latency is high",
    "CPU usage is normal"
  ]
}
```

## 24. Reproducibility

Support:

```text
fixed random seed
configurable model
configurable retrieval strategy
configurable k
configurable dataset split
```

Use YAML configuration.

Example:

```yaml
seed: 42

dataset:
  num_cases: 300

retrieval:
  type: tfidf
  k: 3

model:
  backend: local
  name: qwen-small

evaluation:
  save_raw_outputs: true
```

## 25. Minimal CLI

Provide commands such as:

```bash
python -m context_decisions.cli generate-data

python -m context_decisions.cli run --experiment no-context

python -m context_decisions.cli run --experiment full-context

python -m context_decisions.cli run --experiment retrieval

python -m context_decisions.cli compare
```

## 26. First Milestone

Complete when:

- 200 synthetic cases exist,
- the rule baseline works,
- one small model backend works,
- no-context evaluation works,
- full-context evaluation works,
- retrieval evaluation works,
- results are stored,
- a comparison table is generated.

## 27. Second Milestone

Add:

```text
irrelevant context experiment
missing-information cases
confidence calibration
large-model reference
```

Produce plots for:

```text
accuracy vs context noise
accuracy vs retrieved k
confidence vs correctness
small vs large model latency
```

## 28. Third Milestone

Optional:

```text
fine-tuning
LoRA
information-seeking loop
two-stage escalation
Needle
Jev-like model
multiple domains
```

## 29. Coding Agent Priorities

Implement in this order:

1. repository structure,
2. schemas,
3. synthetic rule system,
4. dataset generator,
5. rule baseline,
6. context builders,
7. evaluation runner,
8. metrics,
9. generic decision-model interface,
10. one model backend,
11. retrieval,
12. experiment scripts,
13. comparison output,
14. tests,
15. documentation.

Avoid premature abstractions.

Prefer:

```text
small readable functions
explicit data flow
plain JSON / JSONL
simple configuration
```

over agent frameworks, workflow engines, distributed components, or vector databases.

## 30. Design Constraints

Preserve this flow:

```text
Dataset
   ↓
Context Strategy
   ↓
Decision Context Builder
   ↓
Decision Model
   ↓
Decision Result
   ↓
Evaluator
```

Requirements:

- retrieval is not embedded inside the model implementation,
- evaluation does not know which model backend is used,
- the model never sees the ground-truth decision,
- the dataset generator is separate from evaluation.

## 31. Definition of Done for V1

V1 is done when the project can empirically answer:

> **Does a small decision model perform better with full context or with retrieved relevant context than with no domain context?**

The architecture should then make it easy to investigate:

> **Does fine-tuning reduce the amount of runtime context required?**

## 32. Expected Output

The project should eventually produce:

```text
dataset
experiment configs
raw result logs
evaluation tables
calibration plots
context robustness plots
latency comparisons
short written analysis
```

The project should remain focused on understanding:

> **where domain knowledge should live in a small-model decision architecture.**

## 33. Design Decisions (V1 implementation)

Decisions taken while implementing phases 0–6, where the plan above left room:

1. **Large fact pool per case.** Every case carries its diagnostic facts plus 10 distractors
   (`dataset.distractors_per_case`). Several distractors share vocabulary with diagnostic facts
   ("CPU usage on the staging cluster is elevated"), so retrieval has to discriminate.
2. **Structured facts in `DecisionContext`.** `DecisionContext.facts` holds only the facts the
   context strategy selected. The rule baseline reads them; LLM backends read `context_text`.
   Cases also store `fact_texts`, the sentence rendering of each fact, fixed at generation time.
3. **No label leakage through the request.** Requests are sampled independently of the
   decision. `generator.request_leakage` checks this (held-out accuracy ≈ chance, 1/7).
4. **Ordered, monotone rules.** Precedence: escalate > rollback > application > database >
   network > infrastructure > ask. Rules fire only on present, positive evidence, so removing
   facts never makes a rule fire. Hence `relevant_facts` (the facts the winning rule tested)
   are sufficient on their own. `escalate` = high customer impact and ≥ 2 degraded subsystems
   (database, network, infrastructure).
5. **Scoring against the full-information decision.** Ground truth never changes with the
   context condition. Abstention is reported separately as `ask_rate_on_answerable`.
6. **Oracle condition.** `OracleContext` shows exactly the relevant facts: the upper bound for
   any retriever, separating retrieval quality from how the model uses context.
7. **Backend.** First model backend is simple-jev's logit-based choice API (full probability
   distribution → Brier/ECE without parsing).
