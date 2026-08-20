import json
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from agent_debugger_evals.agent import AgentResult
from agent_debugger_evals.artifacts import RunArtifactStore
from agent_debugger_evals.models import EvaluationStatus, EvaluationTask, TokenUsage
from agent_debugger_evals.runner import (
    EvaluationRunner,
    EvaluationRunnerError,
    InvalidPatchError,
    PatchScorer,
)
from agent_debugger_evals.sandbox import LocalSandbox
from agent_debugger_evals.tracing import JsonlTracer


def run_git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def initialize_repository(repository: Path) -> str:
    repository.mkdir()
    (repository / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repository / ".gitignore").write_text(".setup-ready\n", encoding="utf-8")
    (repository / "test_base.py").write_text(
        "from app import VALUE\n\n\ndef test_value():\n    assert VALUE == 2\n",
        encoding="utf-8",
    )
    run_git(repository, "init", "--quiet")
    run_git(repository, "add", ".")
    run_git(
        repository,
        "-c",
        "user.name=Test User",
        "-c",
        "user.email=test@example.com",
        "commit",
        "--quiet",
        "-m",
        "Initial commit",
    )
    return run_git(repository, "rev-parse", "HEAD")


class FakeSandbox:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    @contextmanager
    def create(self, task: EvaluationTask) -> Iterator[Path]:
        yield self.workspace


class FakeAgent:
    def __init__(self) -> None:
        self.workspace: Path | None = None

    def solve(self, problem: str, workspace: Path) -> AgentResult:
        self.workspace = workspace
        assert problem == "Set the value to two."
        assert (workspace / ".setup-ready").read_text(encoding="utf-8") == "ready"
        assert not (workspace / ".agent-eval-hidden-tests").exists()
        (workspace / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
        (workspace / "new_file.txt").write_text("created\n", encoding="utf-8")
        return AgentResult(
            run_id="run-123",
            agent_id="agent-123",
            status="finished",
            output="Done",
            duration_seconds=0.5,
            usage=TokenUsage(
                input_tokens=10,
                output_tokens=5,
                cache_read_tokens=2,
                cache_write_tokens=1,
                total_tokens=18,
            ),
            cost_usd=0.01,
        )


def test_runner_persists_attempt_then_scores_fresh_workspace(tmp_path: Path) -> None:
    source = tmp_path / "source"
    commit = initialize_repository(source)
    agent_workspace = tmp_path / "agent-workspace"
    scorer_workspace = tmp_path / "scorer-workspace"
    run_git(tmp_path, "clone", "--quiet", str(source), str(agent_workspace))
    run_git(tmp_path, "clone", "--quiet", str(source), str(scorer_workspace))

    hidden_test = tmp_path / "test_hidden.py"
    hidden_test.write_text(
        "from app import VALUE\n\n\ndef test_hidden_value():\n    assert VALUE == 2\n",
        encoding="utf-8",
    )
    task = EvaluationTask(
        id="runner-test",
        repository=str(source),
        base_commit=commit,
        problem="Set the value to two.",
        setup_command=(
            'python -c "from pathlib import Path; '
            "Path('.setup-ready').write_text('ready')\""
        ),
        test_command="python -m pytest -q test_base.py",
        timeout_seconds=30,
        hidden_tests=(hidden_test,),
    )
    trace_file = tmp_path / "evaluation.jsonl"
    artifact_dir = tmp_path / "artifacts" / "run-123"
    agent = FakeAgent()

    result = EvaluationRunner(
        agent=agent,
        agent_sandbox=FakeSandbox(agent_workspace),
        scorer_sandbox=FakeSandbox(scorer_workspace),
        artifact_store=RunArtifactStore(tmp_path / "artifacts"),
        tracer=JsonlTracer(trace_file),
    ).run(task, artifact_dir)

    assert result.status is EvaluationStatus.PASSED
    assert result.passed
    assert agent.workspace == agent_workspace
    assert agent.workspace != scorer_workspace
    assert (scorer_workspace / "app.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert result.attempt.patch_path.read_text(encoding="utf-8")
    assert result.attempt.agent_output_path.read_text(encoding="utf-8") == "Done"
    assert result.attempt.metadata_path.is_file()
    assert result.attempt.usage is not None
    assert result.attempt.usage.total_tokens == 18
    assert result.score.environment_hash
    assert "2 passed" in result.score.test_output
    assert "test_hidden_value" not in result.attempt.patch_path.read_text(
        encoding="utf-8"
    )

    artifact_events = [
        json.loads(line)
        for line in result.attempt.trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in artifact_events] == [
        "attempt_completed",
        "patch_persisted",
        "scoring_started",
        "scorer_environment_prepared",
        "scoring_completed",
    ]


def test_scoring_boundary_is_repeatable_for_base_and_known_patch(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    commit = initialize_repository(repository)
    hidden_test = tmp_path / "test_hidden.py"
    hidden_test.write_text(
        "from app import VALUE\n\n\ndef test_hidden_value():\n    assert VALUE == 2\n",
        encoding="utf-8",
    )
    task = EvaluationTask(
        id="boundary-test",
        repository=str(repository),
        base_commit=commit,
        problem="Set the value to two.",
        test_command="python -m pytest -q test_base.py",
        timeout_seconds=30,
        hidden_tests=(hidden_test,),
    )
    empty_patch = tmp_path / "empty.diff"
    empty_patch.write_text("", encoding="utf-8")
    good_patch = tmp_path / "good.diff"
    (repository / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    good_patch.write_text(
        run_git(repository, "diff", "--binary", "HEAD") + "\n",
        encoding="utf-8",
    )
    run_git(repository, "checkout", "--", "app.py")
    scorer = PatchScorer(LocalSandbox())

    baseline = scorer.score_patch(task, empty_patch)
    first = scorer.score_patch(task, good_patch)
    second = scorer.score_patch(task, good_patch)

    assert baseline.status is EvaluationStatus.TEST_FAILED
    assert not baseline.passed
    assert first.status is EvaluationStatus.PASSED
    assert first.passed
    assert second.passed
    assert first.exit_code == second.exit_code
    assert first.environment_hash == second.environment_hash


def test_patch_scorer_rejects_invalid_patch(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    commit = initialize_repository(repository)
    task = EvaluationTask(
        id="invalid-patch",
        repository=str(repository),
        base_commit=commit,
        problem="unused",
        test_command="python -m pytest",
        timeout_seconds=30,
    )
    patch = tmp_path / "invalid.diff"
    patch.write_text("not a git patch\n", encoding="utf-8")

    with pytest.raises(InvalidPatchError, match="could not be applied"):
        PatchScorer(LocalSandbox()).score_patch(task, patch)

    protected_file = repository / ".agent-eval-hidden-tests" / "injected.py"
    protected_file.parent.mkdir()
    protected_file.write_text("INJECTED = True\n", encoding="utf-8")
    run_git(repository, "add", "--intent-to-add", str(protected_file))
    protected_patch = tmp_path / "protected.diff"
    protected_patch.write_text(
        run_git(repository, "diff", "--binary", "HEAD") + "\n",
        encoding="utf-8",
    )
    run_git(repository, "reset", "--quiet")
    protected_file.unlink()
    protected_file.parent.rmdir()

    with pytest.raises(InvalidPatchError, match="protected path"):
        PatchScorer(LocalSandbox()).score_patch(task, protected_patch)


def test_patch_scorer_rejects_mismatched_task_metadata(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    patch = artifact_dir / "patch.diff"
    patch.write_text("", encoding="utf-8")
    (artifact_dir / "metadata.json").write_text(
        '{"task_fingerprint": "different"}\n',
        encoding="utf-8",
    )
    task = EvaluationTask(
        id="task-binding",
        repository="unused",
        base_commit="unused",
        problem="unused",
        test_command="pytest",
        timeout_seconds=30,
    )

    with pytest.raises(InvalidPatchError, match="different task definition"):
        PatchScorer(LocalSandbox()).score_patch(task, patch)


def test_runner_marks_timed_out_tests_as_failed(tmp_path: Path) -> None:
    task = EvaluationTask(
        id="timeout-test",
        repository="unused",
        base_commit="unused",
        problem="unused",
        test_command='python -c "import time; time.sleep(1)"',
        timeout_seconds=0.01,
    )

    exit_code, output = EvaluationRunner._run_tests(task, tmp_path, ())

    assert exit_code == 124
    assert "timed out" in output


def test_runner_reports_failed_environment_setup(tmp_path: Path) -> None:
    task = EvaluationTask(
        id="setup-failure",
        repository="unused",
        base_commit="unused",
        problem="unused",
        setup_command='python -c "raise SystemExit(7)"',
        test_command="python -m pytest",
        timeout_seconds=30,
    )

    with pytest.raises(EvaluationRunnerError, match="exited with code 7"):
        EvaluationRunner._prepare_environment(task, tmp_path)
