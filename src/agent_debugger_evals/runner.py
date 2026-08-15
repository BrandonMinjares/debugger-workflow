import shlex
import shutil
import subprocess
import time
from pathlib import Path

from .agent import CodingAgent
from .models import EvaluationResult, EvaluationTask
from .sandbox import Sandbox
from .scoring import score_test_run
from .tracing import JsonlTracer


class EvaluationRunnerError(RuntimeError):
    """Raised when evaluation infrastructure cannot complete a run."""


class EvaluationRunner:
    """Coordinates the agent, sandbox, tracing, and deterministic scorer."""

    def __init__(
        self,
        agent: CodingAgent,
        sandbox: Sandbox,
        tracer: JsonlTracer | None = None,
    ) -> None:
        self.agent = agent
        self.sandbox = sandbox
        self.tracer = tracer

    def run(self, task: EvaluationTask) -> EvaluationResult:
        started_at = time.monotonic()
        self._record(
            "task_started",
            {
                "task_id": task.id,
                "repository": task.repository,
                "base_commit": task.base_commit,
            },
        )

        try:
            with self.sandbox.create(task) as workspace:
                setup_output = self._prepare_environment(task, workspace)
                if task.setup_command is not None:
                    self._record(
                        "environment_prepared",
                        {
                            "task_id": task.id,
                            "command": task.setup_command,
                            "output": setup_output,
                        },
                    )
                agent_result = self.agent.solve(task.problem, workspace)
                self._record("agent_completed", agent_result)
                patch = self._capture_patch(workspace)
                hidden_tests = self._install_hidden_tests(task, workspace)
                exit_code, output = self._run_tests(task, workspace, hidden_tests)

            score = score_test_run(exit_code, output)
            self._record(
                "tests_completed",
                {
                    "task_id": task.id,
                    "passed": score.passed,
                    "exit_code": score.exit_code,
                    "output": score.output,
                },
            )
            evaluation = EvaluationResult(
                task_id=task.id,
                passed=score.passed,
                exit_code=score.exit_code,
                duration_seconds=time.monotonic() - started_at,
                patch=patch,
                test_output=score.output,
                agent_run_id=agent_result.run_id,
                agent_id=agent_result.agent_id,
                agent_status=agent_result.status,
                agent_output=agent_result.output,
                agent_duration_seconds=agent_result.duration_seconds,
                usage=agent_result.usage,
                cost_usd=agent_result.cost_usd,
            )
            self._record("evaluation_completed", evaluation)
            return evaluation
        except Exception as error:
            self._record(
                "evaluation_failed",
                {
                    "task_id": task.id,
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )
            raise

    def _record(self, event: str, payload: object) -> None:
        if self.tracer is not None:
            self.tracer.record(event, payload)

    @staticmethod
    def _prepare_environment(task: EvaluationTask, workspace: Path) -> str:
        if task.setup_command is None:
            return ""

        command = shlex.split(task.setup_command)
        if not command:
            raise EvaluationRunnerError("Task setup command cannot be empty")

        try:
            result = subprocess.run(
                command,
                cwd=workspace,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=task.setup_timeout_seconds,
            )
        except OSError as error:
            raise EvaluationRunnerError(
                f"Could not run setup command: {error}"
            ) from error
        except subprocess.TimeoutExpired as error:
            output = error.stdout or ""
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            raise EvaluationRunnerError(
                f"Setup command timed out after {task.setup_timeout_seconds} "
                f"seconds:\n{output}"
            ) from error

        if result.returncode != 0:
            raise EvaluationRunnerError(
                f"Setup command exited with code {result.returncode}:\n"
                f"{result.stdout}"
            )

        return result.stdout

    @staticmethod
    def _capture_patch(workspace: Path) -> str:
        try:
            subprocess.run(
                ["git", "add", "--intent-to-add", "--all"],
                cwd=workspace,
                check=True,
                capture_output=True,
                text=True,
            )
            result = subprocess.run(
                ["git", "diff", "--binary", "HEAD"],
                cwd=workspace,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as error:
            detail = error.stderr.strip() or str(error)
            raise EvaluationRunnerError(f"Could not capture agent patch: {detail}") from error

        return result.stdout

    @staticmethod
    def _install_hidden_tests(
        task: EvaluationTask, workspace: Path
    ) -> tuple[Path, ...]:
        if not task.hidden_tests:
            return ()

        destination = workspace / ".agent-eval-hidden-tests"
        destination.mkdir()
        installed = []
        for index, source in enumerate(task.hidden_tests):
            if not source.is_file():
                raise EvaluationRunnerError(f"Hidden test does not exist: {source}")
            target = destination / f"{index}_{source.name}"
            shutil.copy2(source, target)
            installed.append(target.relative_to(workspace))

        return tuple(installed)

    @staticmethod
    def _run_tests(
        task: EvaluationTask,
        workspace: Path,
        hidden_tests: tuple[Path, ...],
    ) -> tuple[int, str]:
        command = [
            *shlex.split(task.test_command),
            *(str(path) for path in hidden_tests),
        ]
        if not command:
            raise EvaluationRunnerError("Task test command cannot be empty")

        try:
            result = subprocess.run(
                command,
                cwd=workspace,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=task.timeout_seconds,
            )
            return result.returncode, result.stdout
        except OSError as error:
            raise EvaluationRunnerError(
                f"Could not run test command: {error}"
            ) from error
        except subprocess.TimeoutExpired as error:
            output = error.stdout or ""
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            message = f"Test command timed out after {task.timeout_seconds} seconds"
            return 124, f"{output}\n{message}".lstrip()
