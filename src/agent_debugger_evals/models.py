from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvaluationTask:
    """A reproducible coding-agent task."""

    id: str
    repository: str
    base_commit: str
    problem: str
    test_command: str
    timeout_seconds: int
    hidden_tests: tuple[Path, ...] = ()


@dataclass(frozen=True)
class EvaluationResult:
    """The measurements produced by one agent attempt."""

    task_id: str
    passed: bool
    exit_code: int
    duration_seconds: float
    patch: str
