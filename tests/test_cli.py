from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from agent_debugger_evals.cli import app
from agent_debugger_evals.models import (
    AttemptResult,
    EvaluationResult,
    EvaluationStatus,
    ScoreResult,
    TokenUsage,
)

cli = CliRunner()
task_file = Path("tasks/click/click-3449/task.yaml")


def make_attempt(tmp_path: Path) -> AttemptResult:
    artifact_dir = tmp_path / "run-123"
    return AttemptResult(
        task_id="click-3449",
        run_id="run-123",
        agent_id="agent-123",
        agent_status="finished",
        duration_seconds=2.0,
        artifact_dir=artifact_dir,
        patch_path=artifact_dir / "patch.diff",
        agent_output_path=artifact_dir / "agent-output.txt",
        metadata_path=artifact_dir / "metadata.json",
        trace_path=artifact_dir / "trace.jsonl",
        usage=TokenUsage(
            input_tokens=10,
            output_tokens=5,
            cache_read_tokens=0,
            cache_write_tokens=0,
            total_tokens=15,
        ),
    )


def make_score(tmp_path: Path, passed: bool = True) -> ScoreResult:
    return ScoreResult(
        task_id="click-3449",
        status=(
            EvaluationStatus.PASSED if passed else EvaluationStatus.TEST_FAILED
        ),
        passed=passed,
        exit_code=0 if passed else 1,
        duration_seconds=0.5,
        test_output="passed" if passed else "failed",
        environment_hash="abc123",
        patch_path=tmp_path / "run-123" / "patch.diff",
    )


def test_attempt_command_reports_persisted_patch(tmp_path: Path) -> None:
    attempt = make_attempt(tmp_path)

    with patch(
        "agent_debugger_evals.cli.AttemptRunner.create_attempt",
        return_value=attempt,
    ):
        response = cli.invoke(
            app,
            ["attempt", str(task_file), "--model", "test-model"],
        )

    assert response.exit_code == 0
    assert "ATTEMPT: run-123" in response.output
    assert f"Patch: {attempt.patch_path}" in response.output
    assert "Tokens: 15" in response.output


def test_score_command_does_not_create_agent(tmp_path: Path) -> None:
    score = make_score(tmp_path)
    patch_file = tmp_path / "run-123" / "patch.diff"

    with patch(
        "agent_debugger_evals.cli.PatchScorer.score_patch",
        return_value=score,
    ) as score_patch:
        response = cli.invoke(
            app,
            ["score", str(task_file), str(patch_file)],
        )

    assert response.exit_code == 0
    assert "PASSED: click-3449" in response.output
    assert "Environment: abc123" in response.output
    assert score_patch.call_args.args[1] == patch_file


def test_run_command_reports_composed_success(tmp_path: Path) -> None:
    attempt = make_attempt(tmp_path)
    score = make_score(tmp_path)
    result = EvaluationResult(
        task_id="click-3449",
        status=EvaluationStatus.PASSED,
        attempt=attempt,
        score=score,
    )

    with patch(
        "agent_debugger_evals.cli.EvaluationRunner.run",
        return_value=result,
    ) as run:
        response = cli.invoke(
            app,
            ["run", str(task_file), "--model", "test-model"],
        )

    assert response.exit_code == 0
    assert "PASS: click-3449 (attempt 2.00s, score 0.50s)" in response.output
    assert "Status: passed" in response.output
    assert "Tokens: 15" in response.output
    assert run.call_args.args[0].id == "click-3449"


def test_run_command_exits_nonzero_for_failed_score(tmp_path: Path) -> None:
    result = EvaluationResult(
        task_id="click-3449",
        status=EvaluationStatus.TEST_FAILED,
        attempt=make_attempt(tmp_path),
        score=make_score(tmp_path, passed=False),
    )

    with patch(
        "agent_debugger_evals.cli.EvaluationRunner.run",
        return_value=result,
    ):
        response = cli.invoke(
            app,
            ["run", str(task_file), "--model", "test-model"],
        )

    assert response.exit_code == 1
    assert "FAIL: click-3449" in response.output
