import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .models import (
    EvaluationStatus,
    JudgeLabel,
    JudgeResult,
    ScoreResult,
)

OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
_MAX_PATCH_CHARS = 12_000
_MAX_OUTPUT_CHARS = 8_000
_VALID_LABELS = {label.value for label in JudgeLabel}

_SYSTEM_PROMPT = """\
You classify why a coding agent's patch failed held-out tests.
Return a JSON object with keys "label" and "rationale".
label must be one of: logic_error, wrong_scope, hallucinated_api, \
incomplete_fix, test_only_change, setup_issue, unknown.
rationale must be a short explanation. Do not propose a new patch.
"""


class JudgeError(RuntimeError):
    """Raised when a failure judge cannot produce a classification."""


class FailureJudge(Protocol):
    """Classify a failed coding-agent attempt without changing its score."""

    def classify(
        self,
        problem: str,
        patch: str,
        test_output: str,
        status: EvaluationStatus,
    ) -> JudgeResult:
        """Return a diagnostic label for a failed attempt."""
        ...


@dataclass(frozen=True)
class OpenAIFailureJudge:
    """Call the OpenAI Chat Completions API to classify a failure."""

    model: str = DEFAULT_JUDGE_MODEL
    api_key: str | None = None
    timeout_seconds: float = 60.0

    def classify(
        self,
        problem: str,
        patch: str,
        test_output: str,
        status: EvaluationStatus,
    ) -> JudgeResult:
        key = self.resolved_api_key()
        if not key:
            raise JudgeError("OPENAI_API_KEY must be set to run the failure judge")

        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _user_prompt(problem, patch, test_output, status),
                },
            ],
        }
        response = post_chat_completion(payload, api_key=key, timeout_seconds=self.timeout_seconds)
        return parse_judge_response(response, model=self.model)

    def resolved_api_key(self) -> str | None:
        return self.api_key or os.environ.get("OPENAI_API_KEY")


def post_chat_completion(
    payload: dict[str, Any],
    api_key: str,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """POST a chat-completions request to the OpenAI HTTP API."""
    request = urllib.request.Request(
        OPENAI_CHAT_COMPLETIONS_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise JudgeError(
            f"OpenAI judge request failed with HTTP {error.code}: {detail}"
        ) from error
    except urllib.error.URLError as error:
        raise JudgeError(f"OpenAI judge request failed: {error.reason}") from error
    except json.JSONDecodeError as error:
        raise JudgeError("OpenAI judge returned invalid JSON") from error


def parse_judge_response(response: dict[str, Any], model: str) -> JudgeResult:
    """Parse a chat-completions payload into a constrained JudgeResult."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise JudgeError("OpenAI judge response was missing message content") from error
    if not isinstance(content, str) or not content.strip():
        raise JudgeError("OpenAI judge response was empty")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as error:
        raise JudgeError("OpenAI judge content was not JSON") from error

    raw_label = str(parsed.get("label", "unknown")).strip()
    label = (
        JudgeLabel(raw_label)
        if raw_label in _VALID_LABELS
        else JudgeLabel.UNKNOWN
    )
    rationale = str(parsed.get("rationale", "")).strip() or "No rationale provided."
    return JudgeResult(label=label, rationale=rationale, model=model)


def load_score_from_trace(trace_path: Path) -> ScoreResult:
    """Return the last scoring_completed payload from a run trace."""
    if not trace_path.is_file():
        raise JudgeError(f"Trace file does not exist: {trace_path}")

    last: dict[str, Any] | None = None
    try:
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("event") == "scoring_completed":
                payload = event.get("payload")
                if isinstance(payload, dict):
                    last = payload
    except (OSError, json.JSONDecodeError) as error:
        raise JudgeError(f"Could not read scoring trace: {trace_path}") from error

    if last is None:
        raise JudgeError(
            "No scoring_completed event in the run trace; score the patch first"
        )

    try:
        return ScoreResult(
            task_id=str(last["task_id"]),
            status=EvaluationStatus(last["status"]),
            passed=bool(last["passed"]),
            exit_code=int(last["exit_code"]),
            duration_seconds=float(last["duration_seconds"]),
            test_output=str(last["test_output"]),
            environment_hash=str(last["environment_hash"]),
            patch_path=Path(str(last["patch_path"])),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise JudgeError("scoring_completed payload was incomplete") from error


def _user_prompt(
    problem: str,
    patch: str,
    test_output: str,
    status: EvaluationStatus,
) -> str:
    return (
        f"Status: {status.value}\n\n"
        f"Problem:\n{problem}\n\n"
        f"Patch:\n{_truncate(patch, _MAX_PATCH_CHARS)}\n\n"
        f"Test output:\n{_truncate(test_output, _MAX_OUTPUT_CHARS)}\n"
    )


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...[truncated]\n"
