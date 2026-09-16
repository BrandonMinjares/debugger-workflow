from dataclasses import dataclass
from enum import StrEnum
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


class EvaluationStatus(StrEnum):
    """Terminal classifications for an evaluation."""

    AGENT_ERROR = "agent_error"
    INVALID_PATCH = "invalid_patch"
    TEST_FAILED = "test_failed"
    SCORER_ERROR = "scorer_error"
    PASSED = "passed"


@dataclass(frozen=True)
class AttemptResult:
    """Persisted output from one paid agent attempt."""

    task_id: str
    run_id: str
    agent_id: str
    agent_status: str
    duration_seconds: float
    artifact_dir: Path
    patch_path: Path
    agent_output_path: Path
    metadata_path: Path
    trace_path: Path
    usage: TokenUsage | None = None
    cost_usd: float | None = None


@dataclass(frozen=True)
class ScoreResult:
    """Deterministic result from scoring a persisted patch."""

    task_id: str
    status: EvaluationStatus
    passed: bool
    exit_code: int
    duration_seconds: float
    test_output: str
    environment_hash: str
    patch_path: Path


class JudgeLabel(StrEnum):
    """Fixed failure-mode labels returned by an LLM judge."""

    LOGIC_ERROR = "logic_error"
    WRONG_SCOPE = "wrong_scope"
    HALLUCINATED_API = "hallucinated_api"
    INCOMPLETE_FIX = "incomplete_fix"
    TEST_ONLY_CHANGE = "test_only_change"
    SETUP_ISSUE = "setup_issue"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class JudgeResult:
    """Diagnostic classification that does not affect pass/fail."""

    label: JudgeLabel
    rationale: str
    model: str
    skipped: bool = False


@dataclass(frozen=True)
class EvaluationResult:
    """Combined agent-attempt and independent-scoring result."""

    task_id: str
    status: EvaluationStatus
    attempt: AttemptResult
    score: ScoreResult
    judge: JudgeResult | None = None

    @property
    def passed(self) -> bool:
        return self.status is EvaluationStatus.PASSED
