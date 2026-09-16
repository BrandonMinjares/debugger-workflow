import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_debugger_evals.judge import (
    JudgeError,
    OpenAIFailureJudge,
    load_score_from_trace,
    parse_judge_response,
)
from agent_debugger_evals.models import EvaluationStatus, JudgeLabel
from agent_debugger_evals.tracing import JsonlTracer


def test_parse_judge_response_accepts_known_label() -> None:
    result = parse_judge_response(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "label": "hallucinated_api",
                                "rationale": "Called a missing helper.",
                            }
                        )
                    }
                }
            ]
        },
        model="gpt-4o-mini",
    )

    assert result.label is JudgeLabel.HALLUCINATED_API
    assert result.rationale == "Called a missing helper."
    assert result.model == "gpt-4o-mini"
    assert not result.skipped


def test_parse_judge_response_coerces_unknown_label() -> None:
    result = parse_judge_response(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"label": "not-a-label", "rationale": "unclear"}
                        )
                    }
                }
            ]
        },
        model="gpt-4o-mini",
    )

    assert result.label is JudgeLabel.UNKNOWN


def test_openai_judge_requires_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(JudgeError, match="OPENAI_API_KEY"):
        OpenAIFailureJudge().classify(
            problem="Fix the bug.",
            patch="",
            test_output="failed",
            status=EvaluationStatus.TEST_FAILED,
        )


def test_openai_judge_posts_chat_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "label": "incomplete_fix",
                            "rationale": "Tests still fail.",
                        }
                    )
                }
            }
        ]
    }

    with patch(
        "agent_debugger_evals.judge.post_chat_completion",
        return_value=response,
    ) as post:
        result = OpenAIFailureJudge(model="gpt-4o-mini").classify(
            problem="Set VALUE to 2.",
            patch="diff --git a/app.py",
            test_output="1 failed",
            status=EvaluationStatus.TEST_FAILED,
        )

    payload = post.call_args.args[0]
    assert payload["model"] == "gpt-4o-mini"
    assert payload["response_format"] == {"type": "json_object"}
    assert result.label is JudgeLabel.INCOMPLETE_FIX


def test_load_score_from_trace_reads_last_scoring_event(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    tracer = JsonlTracer(trace_path)
    tracer.record(
        "scoring_completed",
        {
            "task_id": "task-1",
            "status": "test_failed",
            "passed": False,
            "exit_code": 1,
            "duration_seconds": 0.2,
            "test_output": "failed",
            "environment_hash": "abc",
            "patch_path": str(tmp_path / "patch.diff"),
        },
    )

    score = load_score_from_trace(trace_path)

    assert score.task_id == "task-1"
    assert score.status is EvaluationStatus.TEST_FAILED
    assert not score.passed


def test_load_score_from_trace_requires_scoring_event(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps({"event": "attempt_completed", "payload": {}}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(JudgeError, match="score the patch first"):
        load_score_from_trace(trace_path)
