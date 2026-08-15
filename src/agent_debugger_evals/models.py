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
    setup_command: str | None = None
    setup_timeout_seconds: int = 300
    hidden_tests: tuple[Path, ...] = ()


@dataclass(frozen=True)
class TokenUsage:
    """Provider-neutral token usage for one agent attempt."""

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    total_tokens: int
    reasoning_tokens: int | None = None


@dataclass(frozen=True)
class EvaluationResult:
    """The measurements produced by one agent attempt."""

    task_id: str
    passed: bool
    exit_code: int
    duration_seconds: float
    patch: str
    test_output: str = ""
    agent_run_id: str | None = None
    agent_id: str | None = None
    agent_status: str | None = None
    agent_output: str = ""
    agent_duration_seconds: float = 0.0
    usage: TokenUsage | None = None
    cost_usd: float | None = None
