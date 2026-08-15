import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cursor_sdk import Agent, AgentOptions, CursorAgentError, LocalAgentOptions

from .models import TokenUsage


@dataclass(frozen=True)
class AgentResult:
    """Metadata returned after a coding-agent attempt."""

    run_id: str
    agent_id: str
    status: str
    output: str
    duration_seconds: float
    usage: TokenUsage | None = None
    cost_usd: float | None = None


class AgentExecutionError(RuntimeError):
    """Raised when an agent run cannot be started."""


class CodingAgent(Protocol):
    """Provider-neutral interface for an agent under evaluation."""

    def solve(self, problem: str, workspace: Path) -> AgentResult:
        """Attempt to solve a problem by modifying the workspace."""
        ...


class CursorCodingAgent:
    """Run a one-shot Cursor agent against a local task workspace."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("CURSOR_API_KEY")

    def solve(self, problem: str, workspace: Path) -> AgentResult:
        if not self.api_key:
            raise AgentExecutionError(
                "CURSOR_API_KEY must be set to run a Cursor coding agent"
            )
        if not workspace.is_dir():
            raise AgentExecutionError(f"Agent workspace does not exist: {workspace}")

        prompt = (
            "Solve the coding task below in the current workspace. "
            "Inspect the repository, modify the necessary files, and run relevant "
            "tests. Do not only describe the solution; implement it.\n\n"
            f"{problem}"
        )
        options = AgentOptions(
            api_key=self.api_key,
            model=self.model,
            local=LocalAgentOptions(cwd=workspace),
        )

        try:
            result = Agent.prompt(prompt, options)
        except CursorAgentError as error:
            raise AgentExecutionError(
                f"Cursor agent could not start: {error}"
            ) from error

        usage = None
        if result.usage is not None:
            usage = TokenUsage(
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
                cache_read_tokens=result.usage.cache_read_tokens,
                cache_write_tokens=result.usage.cache_write_tokens,
                total_tokens=result.usage.total_tokens,
                reasoning_tokens=result.usage.reasoning_tokens,
            )

        return AgentResult(
            run_id=result.id,
            agent_id=result.agent_id,
            status=result.status,
            output=result.result,
            duration_seconds=result.duration_ms / 1000,
            usage=usage,
        )
