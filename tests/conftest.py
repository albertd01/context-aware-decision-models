import pytest

from context_decisions.generator import generate

CFG = {"num_cases": 200, "distractors_per_case": 10, "hide_background_prob": 0.3,
       "hard_case_prob": 0.3}


@pytest.fixture(scope="session")
def cases():
    return generate(CFG, seed=42)
