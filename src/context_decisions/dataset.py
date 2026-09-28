"""Plain JSONL storage for cases and run records, plus YAML config loading."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from context_decisions.schemas import DecisionCase


def load_config(path: str | Path = "config/default.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def save_jsonl(records: list[BaseModel], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.model_dump_json() + "\n" for r in records))


def load_cases(path: str | Path) -> list[DecisionCase]:
    lines = Path(path).read_text().splitlines()
    return [DecisionCase.model_validate_json(line) for line in lines if line.strip()]
