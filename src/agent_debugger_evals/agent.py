from pathlib import Path
from typing import Protocol


class CodingAgent(Protocol):
    """Provider-neutral interface for an agent under evaluation."""

    def solve(self, problem: str, workspace: Path) -> None:
        """Attempt to solve a problem by modifying the workspace."""
        ...
