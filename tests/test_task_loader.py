from pathlib import Path

from agent_debugger_evals.task_loader import load_task


def test_load_click_task() -> None:
    task = load_task(Path("tasks/click/click-3449/task.yaml"))

    assert task.id == "click-3449"
    assert task.base_commit == "a5f5aa6d4012d256ccca24638f2642fc371e9f77"
    assert "I/O operation on closed file" in task.problem
    assert task.setup_command == "uv sync"
    assert task.setup_timeout_seconds == 300
    assert task.test_command.startswith("uv run pytest")
    assert len(task.hidden_tests) == 1
