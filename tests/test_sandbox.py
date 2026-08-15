import subprocess
from pathlib import Path

from agent_debugger_evals.models import EvaluationTask
from agent_debugger_evals.sandbox import LocalSandbox


def run_git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_local_sandbox_checks_out_commit_and_cleans_up(tmp_path: Path) -> None:
    repository = tmp_path / "source"
    repository.mkdir()
    run_git(repository, "init", "--quiet")
    (repository / "example.txt").write_text("original\n", encoding="utf-8")
    run_git(repository, "add", "example.txt")
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
    commit = run_git(repository, "rev-parse", "HEAD")
    task = EvaluationTask(
        id="sandbox-test",
        repository=str(repository),
        base_commit=commit,
        problem="Change the example.",
        test_command="true",
        timeout_seconds=30,
    )

    with LocalSandbox().create(task) as workspace:
        sandbox_path = workspace
        assert run_git(workspace, "rev-parse", "HEAD") == commit
        assert (workspace / "example.txt").read_text(encoding="utf-8") == "original\n"
        (workspace / "example.txt").write_text("changed\n", encoding="utf-8")

    assert not sandbox_path.exists()
