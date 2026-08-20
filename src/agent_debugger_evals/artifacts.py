import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .agent import AgentResult
from .models import AttemptResult, EvaluationStatus, EvaluationTask
from .tracing import JsonlTracer


class ArtifactError(RuntimeError):
    """Raised when immutable run artifacts cannot be persisted."""

    status = EvaluationStatus.SCORER_ERROR


def task_fingerprint(task: EvaluationTask) -> str:
    """Hash the immutable task inputs used by an attempt or score."""
    value = {
        "id": task.id,
        "repository": task.repository,
        "base_commit": task.base_commit,
        "problem": task.problem,
        "setup_command": task.setup_command,
        "setup_timeout_seconds": task.setup_timeout_seconds,
        "test_command": task.test_command,
        "timeout_seconds": task.timeout_seconds,
        "hidden_tests": [
            {
                "name": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in task.hidden_tests
        ],
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class RunArtifactStore:
    """Persist immutable outputs from paid agent attempts."""

    def __init__(self, root: Path = Path("artifacts/runs")) -> None:
        self.root = root

    def save(
        self,
        task: EvaluationTask,
        agent_result: AgentResult,
        patch: str,
        output_dir: Path | None = None,
        duration_seconds: float | None = None,
    ) -> AttemptResult:
        artifact_dir = output_dir or self.root / agent_result.run_id
        try:
            artifact_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise ArtifactError(
                f"Artifact directory already exists: {artifact_dir}"
            ) from error

        patch_path = artifact_dir / "patch.diff"
        output_path = artifact_dir / "agent-output.txt"
        metadata_path = artifact_dir / "metadata.json"
        trace_path = artifact_dir / "trace.jsonl"

        patch_path.write_text(patch, encoding="utf-8")
        output_path.write_text(agent_result.output, encoding="utf-8")
        metadata = {
            "schema_version": 1,
            "task_id": task.id,
            "task_fingerprint": task_fingerprint(task),
            "repository": task.repository,
            "base_commit": task.base_commit,
            "agent": asdict(agent_result),
            "files": {
                "patch": patch_path.name,
                "agent_output": output_path.name,
                "trace": trace_path.name,
            },
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

        tracer = JsonlTracer(trace_path)
        tracer.record(
            "attempt_completed",
            {
                "task_id": task.id,
                "task_fingerprint": metadata["task_fingerprint"],
                "agent": asdict(agent_result),
            },
        )
        tracer.record(
            "patch_persisted",
            {
                "patch_path": str(patch_path),
                "size_bytes": patch_path.stat().st_size,
            },
        )

        return AttemptResult(
            task_id=task.id,
            run_id=agent_result.run_id,
            agent_id=agent_result.agent_id,
            agent_status=agent_result.status,
            duration_seconds=(
                duration_seconds
                if duration_seconds is not None
                else agent_result.duration_seconds
            ),
            artifact_dir=artifact_dir,
            patch_path=patch_path,
            agent_output_path=output_path,
            metadata_path=metadata_path,
            trace_path=trace_path,
            usage=agent_result.usage,
            cost_usd=agent_result.cost_usd,
        )
