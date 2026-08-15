from pathlib import Path
from unittest.mock import patch

import pytest
from cursor_sdk import RunResult, TokenUsage

from agent_debugger_evals.agent import AgentExecutionError, CursorCodingAgent


def test_cursor_agent_runs_in_workspace_and_reports_usage(tmp_path: Path) -> None:
    sdk_result = RunResult(
        id="run-123",
        agent_id="agent-123",
        status="finished",
        result="Implemented the fix.",
        duration_ms=1250,
        usage=TokenUsage(
            input_tokens=100,
            output_tokens=20,
            cache_read_tokens=10,
            cache_write_tokens=5,
            total_tokens=135,
            reasoning_tokens=3,
        ),
    )

    with patch(
        "agent_debugger_evals.agent.Agent.prompt", return_value=sdk_result
    ) as prompt:
        result = CursorCodingAgent("test-model", api_key="cursor_test").solve(
            "Fix the bug.",
            tmp_path,
        )

    message, options = prompt.call_args.args
    assert "Fix the bug." in message
    assert options.model == "test-model"
    assert options.api_key == "cursor_test"
    assert options.local is not None
    assert options.local.cwd == tmp_path
    assert result.status == "finished"
    assert result.duration_seconds == 1.25
    assert result.usage is not None
    assert result.usage.total_tokens == 135


def test_cursor_agent_requires_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)

    with pytest.raises(AgentExecutionError, match="CURSOR_API_KEY"):
        CursorCodingAgent("test-model").solve("Fix the bug.", tmp_path)
