from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol

from .models import EvaluationTask


class Sandbox(Protocol):
    """Isolated checkout in which an agent can modify and execute code."""

    def create(self, task: EvaluationTask) -> AbstractContextManager[Path]:
        """Create a disposable workspace pinned to the task's base commit."""
        ...
