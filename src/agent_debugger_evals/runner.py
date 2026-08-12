from .agent import CodingAgent
from .models import EvaluationResult, EvaluationTask
from .sandbox import Sandbox


class EvaluationRunner:
    """Coordinates the agent, sandbox, tracing, and deterministic scorer."""

    def __init__(self, agent: CodingAgent, sandbox: Sandbox) -> None:
        self.agent = agent
        self.sandbox = sandbox

    def run(self, task: EvaluationTask) -> EvaluationResult:
        raise NotImplementedError(
            "The first milestone is implementing one complete run."
        )
