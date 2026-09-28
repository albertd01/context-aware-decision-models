# Context-Aware Decision Models

A small research codebase for testing how **small decision models** use domain knowledge.

The project focuses on a simple question:

> **How should domain knowledge be provided to small System-One-style decision models: through runtime context, retrieval, or model weights?**

The first version uses a synthetic **DevOps incident triage** domain so that all ground-truth decisions are fully controlled and independent of any AKH data or infrastructure.

## Project Goals

The project compares decision quality under different knowledge conditions:

1. **No Context**
2. **Full Context Injection**
3. **Retrieved Context**
4. **Optional Fine-Tuning / Adapter**
5. **Large Reasoning Model Reference**

The main metrics are decision accuracy, macro F1, per-class precision and recall, confidence calibration, latency, robustness to irrelevant context, and abstention / clarification behavior.

The project is intentionally **not** an agent framework. It is an experimental framework for understanding where domain knowledge should live in a small-model decision architecture.

## Quickstart

```bash
uv sync
export PYTHONPATH=src   # see note below

python -m context_decisions.cli generate-data                        # 200 seeded cases
python -m context_decisions.cli run --experiment all --model rules   # rule baseline
python -m context_decisions.cli run --experiment all --model qwen-4b # simple-jev demo API
python -m context_decisions.cli compare                              # results/comparison.md
uv run pytest
```

Experiments: `no-context`, `full-context`, `oracle` (only the relevant facts), `retrieval`
(TF-IDF top-k on the request), `retrieval-expanded` (TF-IDF with domain symptom vocabulary
added to the query). Model presets live in `config/default.yaml`: `rules`, `qwen-4b` and
`qwen-27b` (public simple-jev demo API, 2 requests/s), and `local` (a local `simple-jev` server).

Note: on macOS, folders synced from `~/Desktop` can get the `hidden` flag applied to `.venv`
contents; Python 3.12+ then skips the editable-install `.pth` file and the package fails to
import. `PYTHONPATH=src` side-steps this; pytest already sets it via `pyproject.toml`.

## Initial Decision Domain

The first test domain is synthetic DevOps incident triage.

Example request:

```text
"The API became slow after the latest deployment."
```

Example context:

```text
deployment_recent = true
database_latency = normal
cpu_usage = normal
application_errors = schema mismatch
```

Possible decision:

```text
inspect_application
```

Initial decision space:

```text
inspect_application
inspect_database
inspect_network
inspect_infrastructure
rollback_deployment
ask_for_more_information
escalate
```

## Core Architecture

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

Keep these components independent:

- retrieval must not be hidden inside the model implementation,
- evaluation must not depend on one specific model backend,
- the model must not see ground truth,
- dataset generation must remain separate from evaluation.

## Planned Context Conditions

### No Context

```text
request
→ decision model
```

### Full Context

```text
request
+ all available facts
→ decision model
```

### Retrieved Context

```text
request
→ retriever
→ top-k relevant facts
→ decision model
```

Later:

### Fine-Tuned Model

```text
request + reduced context
→ adapted decision model
```

## Suggested Repository Structure

```text
context-aware-decisions/
│
├── README.md
├── IMPLEMENTATION_PLAN.md
├── pyproject.toml
│
├── config/
│   └── default.yaml
│
├── data/
│   ├── generated/
│   └── domain_knowledge/
│
├── src/
│   └── context_decisions/
│       ├── __init__.py
│       ├── schemas.py
│       ├── dataset.py
│       ├── generator.py
│       ├── context/
│       │   ├── __init__.py
│       │   ├── no_context.py
│       │   ├── full_context.py
│       │   ├── retriever.py
│       │   └── context_builder.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── rule_baseline.py
│       │   ├── llm_decision_model.py
│       │   └── large_model_reference.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   ├── calibration.py
│       │   └── runner.py
│       └── cli.py
│
├── experiments/
│   ├── experiment_01_no_context.py
│   ├── experiment_02_full_context.py
│   ├── experiment_03_retrieval.py
│   └── compare_results.py
│
└── tests/
    ├── test_dataset.py
    ├── test_context_builder.py
    └── test_metrics.py
```

## V1 Research Question

The first version should be able to answer:

> **Does a small decision model perform better with full context or with retrieved relevant context than with no domain context?**

The codebase should then make it easy to investigate:

> **Does fine-tuning reduce the amount of runtime context required?**

## First Milestone

The first milestone is complete when:

- 200 synthetic cases exist,
- the deterministic rule baseline works,
- one small model backend works,
- no-context evaluation works,
- full-context evaluation works,
- retrieved-context evaluation works,
- results are stored as JSONL,
- and a comparison table is generated.

Example output:

```text
Condition          Accuracy   Macro F1   Latency
------------------------------------------------
No Context         0.61       0.58       48 ms
Full Context       0.79       0.76       53 ms
Retrieved Context  0.86       0.83       51 ms
```

## Non-Goals for V1

Do not start with:

- agent frameworks,
- multi-agent orchestration,
- vector databases,
- complex workflow engines,
- real production incident data,
- fine-tuning,
- Jev-specific integration,
- Needle-specific integration,
- information-seeking loops,
- distributed inference.

## Development Philosophy

Prefer:

```text
small readable functions
explicit data flow
plain JSON / JSONL
simple configuration
reproducible experiments
```

over premature abstractions and complex orchestration.

## Next Step

See [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md) for the concrete coding order, schemas, interfaces, experiments, evaluation metrics, and definition of done.
