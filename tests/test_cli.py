from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from agent_debugger_evals.cli import app
from agent_debugger_evals.models import EvaluationResult, TokenUsage

cli = CliRunner()
task_file = Path("tasks/click/click-3449/task.yaml")


def test_run_command_reports_success_and_usage(tmp_path: Path) -> None:
    result = EvaluationResult(
        task_id="click-3449",
        passed=True,
        exit_code=0,
        duration_seconds=2.5,
        patch="diff",
        usage=TokenUsage(
            input_tokens=10,
            output_tokens=5,
            cache_read_tokens=0,
            cache_write_tokens=0,
            total_tokens=15,
        ),
    )
    trace_file = tmp_path / "trace.jsonl"

    with patch(
        "agent_debugger_evals.cli.EvaluationRunner.run",
        return_value=result,
    ) as run:
        response = cli.invoke(
            app,
            [
                "run",
                str(task_file),
                "--model",
                "test-model",
                "--trace-file",
                str(trace_file),
            ],
        )

    assert response.exit_code == 0
    assert "PASS: click-3449 in 2.50s" in response.output
    assert "Tokens: 15" in response.output
    assert f"Trace: {trace_file}" in response.output
    assert run.call_args.args[0].id == "click-3449"


def test_run_command_exits_nonzero_for_failed_evaluation() -> None:
    result = EvaluationResult(
        task_id="click-3449",
        passed=False,
        exit_code=1,
        duration_seconds=1.0,
        patch="",
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
