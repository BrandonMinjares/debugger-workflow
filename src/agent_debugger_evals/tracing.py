import json
from dataclasses import asdict
from pathlib import Path
from typing import Any


class JsonlTracer:
    """Append structured evaluation events to a JSONL trace."""

    def __init__(self, output: Path) -> None:
        self.output = output

    def record(self, event: str, payload: Any) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        value = asdict(payload) if hasattr(payload, "__dataclass_fields__") else payload
        with self.output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": event, "payload": value}, default=str))
            stream.write("\n")
