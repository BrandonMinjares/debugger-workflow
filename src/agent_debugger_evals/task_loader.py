from pathlib import Path

import yaml

from .models import EvaluationTask


def load_task(path: Path) -> EvaluationTask:
    """Load a task manifest while resolving paths relative to the manifest."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    problem_path = path.parent / data["problem_file"]
    hidden_tests = tuple(path.parent / item for item in data.get("hidden_tests", ()))

    return EvaluationTask(
        id=data["id"],
        repository=data["repository"],
        base_commit=data["base_commit"],
        problem=problem_path.read_text(encoding="utf-8"),
        test_command=data["test_command"],
        timeout_seconds=data["timeout_seconds"],
        hidden_tests=hidden_tests,
    )
