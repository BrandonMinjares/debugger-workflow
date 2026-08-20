import json
from pathlib import Path

import pytest

from agent_debugger_evals.agent import AgentResult
from agent_debugger_evals.artifacts import ArtifactError, RunArtifactStore
from agent_debugger_evals.models import EvaluationTask


def test_artifact_store_persists_immutable_attempt_files(tmp_path: Path) -> None:
    task = EvaluationTask(
        id="artifact-test",
        repository="https://example.com/repository.git",
        base_commit="abc123",
        problem="Fix the bug.",
        test_command="pytest",
        timeout_seconds=30,
    )
    agent_result = AgentResult(
        run_id="run-123",
        agent_id="agent-123",
        status="finished",
        output="Implemented the fix.",
        duration_seconds=1.0,
    )
    store = RunArtifactStore(tmp_path / "runs")

    attempt = store.save(task, agent_result, "diff contents\n")

    assert attempt.patch_path.read_text(encoding="utf-8") == "diff contents\n"
    assert (
        attempt.agent_output_path.read_text(encoding="utf-8")
        == "Implemented the fix."
    )
    metadata = json.loads(attempt.metadata_path.read_text(encoding="utf-8"))
    assert metadata["task_id"] == "artifact-test"
    assert metadata["task_fingerprint"]

    with pytest.raises(ArtifactError, match="already exists"):
        store.save(task, agent_result, "replacement\n")

    assert attempt.patch_path.read_text(encoding="utf-8") == "diff contents\n"
