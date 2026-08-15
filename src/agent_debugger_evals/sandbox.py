import subprocess
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

from .models import EvaluationTask


class SandboxError(RuntimeError):
    """Raised when a disposable task workspace cannot be created."""


class Sandbox(Protocol):
    """Isolated checkout in which an agent can modify and execute code."""

    def create(self, task: EvaluationTask) -> AbstractContextManager[Path]:
        """Create a disposable workspace pinned to the task's base commit."""
        ...


class LocalSandbox:
    """A disposable local checkout containing only the requested commit."""

    def __init__(self, setup_timeout_seconds: int = 300) -> None:
        self.setup_timeout_seconds = setup_timeout_seconds

    @contextmanager
    def create(self, task: EvaluationTask) -> Iterator[Path]:
        with TemporaryDirectory(prefix=f"agent-eval-{task.id}-") as directory:
            workspace = Path(directory)
            try:
                self._git("init", "--quiet", cwd=workspace)
                self._git("remote", "add", "origin", task.repository, cwd=workspace)
                self._git(
                    "fetch",
                    "--quiet",
                    "--depth=1",
                    "--filter=blob:none",
                    "origin",
                    task.base_commit,
                    cwd=workspace,
                )
                self._git(
                    "checkout",
                    "--quiet",
                    "--detach",
                    "FETCH_HEAD",
                    cwd=workspace,
                )
            except subprocess.TimeoutExpired as error:
                raise SandboxError(
                    f"Timed out creating sandbox for task {task.id!r}"
                ) from error
            except subprocess.CalledProcessError as error:
                detail = error.stderr.strip() or str(error)
                raise SandboxError(
                    f"Could not create sandbox for task {task.id!r}: {detail}"
                ) from error

            yield workspace

    def _git(self, *arguments: str, cwd: Path) -> None:
        subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=self.setup_timeout_seconds,
        )
