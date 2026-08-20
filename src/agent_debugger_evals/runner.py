import hashlib
import json
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from .agent import CodingAgent
from .artifacts import RunArtifactStore, task_fingerprint
from .models import (
    AttemptResult,
    EvaluationResult,
    EvaluationStatus,
    EvaluationTask,
    ScoreResult,
)
from .sandbox import Sandbox
from .scoring import score_test_run
from .tracing import JsonlTracer


class EvaluationRunnerError(RuntimeError):
    """Raised when evaluation infrastructure cannot complete a run."""

    status = EvaluationStatus.SCORER_ERROR


class InvalidPatchError(EvaluationRunnerError):
    """Raised when a persisted patch is unsafe or cannot be applied."""

    status = EvaluationStatus.INVALID_PATCH


class EvaluationRunner:
    """Compose a paid attempt with independent fresh-checkout scoring."""

    def __init__(
        self,
        agent: CodingAgent,
        agent_sandbox: Sandbox,
        scorer_sandbox: Sandbox,
        artifact_store: RunArtifactStore | None = None,
        tracer: JsonlTracer | None = None,
    ) -> None:
        self.attempt_runner = AttemptRunner(
            agent,
            agent_sandbox,
            artifact_store or RunArtifactStore(),
            tracer,
        )
        self.patch_scorer = PatchScorer(scorer_sandbox)
        self.tracer = tracer

    def run(
        self,
        task: EvaluationTask,
        output_dir: Path | None = None,
    ) -> EvaluationResult:
        self._record(
            "task_started",
            {
                "task_id": task.id,
                "repository": task.repository,
                "base_commit": task.base_commit,
            },
        )

        try:
            attempt = self.attempt_runner.create_attempt(task, output_dir)
            score = self.patch_scorer.score_patch(
                task,
                attempt.patch_path,
                JsonlTracer(attempt.trace_path),
            )
            evaluation = EvaluationResult(
                task_id=task.id,
                status=(
                    EvaluationStatus.AGENT_ERROR
                    if attempt.agent_status != "finished"
                    else score.status
                ),
                attempt=attempt,
                score=score,
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


class AttemptRunner:
    """Run an agent once and persist its patch before cleanup."""

    def __init__(
        self,
        agent: CodingAgent,
        sandbox: Sandbox,
        artifact_store: RunArtifactStore,
        tracer: JsonlTracer | None = None,
    ) -> None:
        self.agent = agent
        self.sandbox = sandbox
        self.artifact_store = artifact_store
        self.tracer = tracer

    def create_attempt(
        self,
        task: EvaluationTask,
        output_dir: Path | None = None,
    ) -> AttemptResult:
        started_at = time.monotonic()
        self._record("attempt_started", {"task_id": task.id})

        with self.sandbox.create(task) as workspace:
            setup_output = EvaluationRunner._prepare_environment(task, workspace)
            if task.setup_command is not None:
                self._record(
                    "agent_environment_prepared",
                    {
                        "task_id": task.id,
                        "command": task.setup_command,
                        "output": setup_output,
                    },
                )
            agent_result = self.agent.solve(task.problem, workspace)
            self._record("agent_completed", agent_result)
            patch = EvaluationRunner._capture_patch(workspace)
            attempt = self.artifact_store.save(
                task,
                agent_result,
                patch,
                output_dir,
                duration_seconds=time.monotonic() - started_at,
            )

        self._record("attempt_persisted", attempt)
        return attempt

    def _record(self, event: str, payload: object) -> None:
        if self.tracer is not None:
            self.tracer.record(event, payload)


class PatchScorer:
    """Apply a persisted patch to a fresh checkout and score it."""

    _protected_paths = (".git", ".agent-eval-hidden-tests")
    _environment_files = (
        ".python-version",
        "Pipfile.lock",
        "poetry.lock",
        "pyproject.toml",
        "requirements.txt",
        "uv.lock",
    )

    def __init__(self, sandbox: Sandbox) -> None:
        self.sandbox = sandbox

    def score_patch(
        self,
        task: EvaluationTask,
        patch_path: Path,
        tracer: JsonlTracer | None = None,
    ) -> ScoreResult:
        started_at = time.monotonic()
        self._record(tracer, "scoring_started", {"task_id": task.id})
        self._validate_task_binding(task, patch_path)

        with self.sandbox.create(task) as workspace:
            setup_output = EvaluationRunner._prepare_environment(task, workspace)
            environment_hash = self._environment_hash(task, workspace)
            self._record(
                tracer,
                "scorer_environment_prepared",
                {
                    "task_id": task.id,
                    "command": task.setup_command,
                    "output": setup_output,
                    "environment_hash": environment_hash,
                },
            )
            self._apply_patch(workspace, patch_path)
            hidden_tests = EvaluationRunner._install_hidden_tests(task, workspace)
            exit_code, output = EvaluationRunner._run_tests(
                task,
                workspace,
                hidden_tests,
            )

        score = score_test_run(exit_code, output)
        result = ScoreResult(
            task_id=task.id,
            status=(
                EvaluationStatus.PASSED
                if score.passed
                else EvaluationStatus.TEST_FAILED
            ),
            passed=score.passed,
            exit_code=score.exit_code,
            duration_seconds=time.monotonic() - started_at,
            test_output=score.output,
            environment_hash=environment_hash,
            patch_path=patch_path,
        )
        self._record(tracer, "scoring_completed", result)
        return result

    @staticmethod
    def _validate_task_binding(
        task: EvaluationTask,
        patch_path: Path,
    ) -> None:
        metadata_path = patch_path.parent / "metadata.json"
        if not metadata_path.is_file():
            return
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise InvalidPatchError(
                f"Could not read patch metadata: {metadata_path}"
            ) from error
        expected = task_fingerprint(task)
        if metadata.get("task_fingerprint") != expected:
            raise InvalidPatchError(
                "Patch artifact was created for a different task definition"
            )

    @classmethod
    def _apply_patch(cls, workspace: Path, patch_path: Path) -> None:
        if not patch_path.is_file():
            raise InvalidPatchError(f"Patch does not exist: {patch_path}")
        if patch_path.stat().st_size == 0:
            return

        for arguments in (
            [
                "git",
                "apply",
                "--check",
                "--index",
                "--binary",
                str(patch_path.resolve()),
            ],
            [
                "git",
                "apply",
                "--index",
                "--binary",
                str(patch_path.resolve()),
            ],
        ):
            try:
                subprocess.run(
                    arguments,
                    cwd=workspace,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as error:
                detail = error.stderr.strip() or str(error)
                raise InvalidPatchError(f"Patch could not be applied: {detail}") from error

        changed = subprocess.run(
            ["git", "diff", "--name-only", "-z", "HEAD"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split("\0")
        for path in filter(None, changed):
            if path in cls._protected_paths or path.startswith(
                tuple(f"{prefix}/" for prefix in cls._protected_paths)
            ):
                raise InvalidPatchError(f"Patch modifies protected path: {path}")

    @classmethod
    def _environment_hash(
        cls,
        task: EvaluationTask,
        workspace: Path,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(task_fingerprint(task).encode())
        for name in cls._environment_files:
            path = workspace / name
            if path.is_file():
                digest.update(name.encode())
                digest.update(path.read_bytes())
        return digest.hexdigest()

    @staticmethod
    def _record(
        tracer: JsonlTracer | None,
        event: str,
        payload: object,
    ) -> None:
        if tracer is not None:
            tracer.record(event, payload)
