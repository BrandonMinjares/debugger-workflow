import json
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from agent_debugger_evals.agent import AgentResult
from agent_debugger_evals.models import EvaluationTask, TokenUsage
from agent_debugger_evals.runner import EvaluationRunner, EvaluationRunnerError
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


class FakeSandbox:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    @contextmanager
    def create(self, task: EvaluationTask) -> Iterator[Path]:
        yield self.workspace


class FakeAgent:
    def solve(self, problem: str, workspace: Path) -> AgentResult:
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


def test_runner_scores_agent_patch_with_hidden_tests(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (workspace / ".gitignore").write_text(".setup-ready\n", encoding="utf-8")
    (workspace / "test_base.py").write_text(
        "from app import VALUE\n\n\ndef test_value():\n    assert VALUE == 2\n",
        encoding="utf-8",
    )
    run_git(workspace, "init", "--quiet")
    run_git(workspace, "add", ".")
    run_git(
        workspace,
        "-c",
        "user.name=Test User",
        "-c",
        "user.email=test@example.com",
        "commit",
        "--quiet",
        "-m",
        "Initial commit",
    )

    hidden_test = tmp_path / "test_hidden.py"
    hidden_test.write_text(
        "from app import VALUE\n\n\ndef test_hidden_value():\n    assert VALUE == 2\n",
        encoding="utf-8",
    )
    task = EvaluationTask(
        id="runner-test",
        repository="unused",
        base_commit=run_git(workspace, "rev-parse", "HEAD"),
        problem="Set the value to two.",
        setup_command=(
            'python -c "from pathlib import Path; '
            "Path('.setup-ready').write_text('ready')\""
        ),
        test_command="python -m pytest -q test_base.py",
        timeout_seconds=30,
        hidden_tests=(hidden_test,),
    )

    trace_file = tmp_path / "trace.jsonl"
    result = EvaluationRunner(
        FakeAgent(),
        FakeSandbox(workspace),
        JsonlTracer(trace_file),
    ).run(task)

    assert result.passed
    assert result.exit_code == 0
    assert result.agent_run_id == "run-123"
    assert result.agent_status == "finished"
    assert result.usage is not None
    assert result.usage.total_tokens == 18
    assert result.cost_usd == 0.01
    assert "2 passed" in result.test_output
    assert "+VALUE = 2" in result.patch
    assert "new_file.txt" in result.patch
    assert "test_hidden_value" not in result.patch
    events = [
        json.loads(line)
        for line in trace_file.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "task_started",
        "environment_prepared",
        "agent_completed",
        "tests_completed",
        "evaluation_completed",
    ]
    assert events[-1]["payload"]["usage"]["total_tokens"] == 18


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
