import json
from pathlib import Path

from agent_debugger_evals.models import EvaluationStatus, ScoreResult
from agent_debugger_evals.tracing import JsonlTracer


def test_jsonl_tracer_appends_serialized_dataclasses(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "trace.jsonl"
    tracer = JsonlTracer(output)
    result = ScoreResult(
        task_id="task-1",
        status=EvaluationStatus.PASSED,
        passed=True,
        exit_code=0,
        duration_seconds=1.5,
        test_output="passed",
        environment_hash="abc123",
        patch_path=tmp_path / "patch.diff",
    )

    tracer.record("evaluation_completed", result)
    tracer.record("note", {"message": "done"})

    events = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert events[0]["event"] == "evaluation_completed"
    assert events[0]["payload"]["task_id"] == "task-1"
    assert events[1] == {"event": "note", "payload": {"message": "done"}}
