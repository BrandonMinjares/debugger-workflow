from dataclasses import dataclass


@dataclass(frozen=True)
class TestScore:
    """Deterministic score derived from the task's test command."""

    passed: bool
    exit_code: int
    output: str


def score_test_run(exit_code: int, output: str) -> TestScore:
    return TestScore(passed=exit_code == 0, exit_code=exit_code, output=output)
