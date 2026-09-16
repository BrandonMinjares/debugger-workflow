# Agent Debugger Evals

A reproducible evaluation and debugging harness for coding agents. The first
benchmark task recreates [Click issue #3449](https://github.com/pallets/click/issues/3449)
from a pinned pre-fix commit and scores the resulting patch with held-out tests.

## Structure

```text
tasks/                       Curated repository tasks and held-out tests
src/agent_debugger_evals/
  agent.py                   Provider-neutral coding-agent interface
  artifacts.py               Immutable run artifact persistence
  sandbox.py                 Isolated checkout interface
  runner.py                  Attempt generation and independent scoring
  scoring.py                 Deterministic test scoring
  judge.py                   Optional LLM failure classification
  tracing.py                 Structured JSONL traces
dashboard/                   Future trace and regression UI
tests/                       Harness unit tests
```

## Setup

```bash
uv sync
export CURSOR_API_KEY="cursor_..."
uv run pytest
uv run agent-eval inspect tasks/click/click-3449/task.yaml
```

## Run an evaluation

The composed command creates a paid agent attempt, persists its patch, destroys
the agent workspace, then scores that patch in a fresh checkout:

```bash
uv run agent-eval run tasks/click/click-3449/task.yaml --model "your-model-id"
```

Generation and scoring can also run separately. Rescoring never calls the model:

```bash
uv run agent-eval attempt tasks/click/click-3449/task.yaml \
  --model "your-model-id" --output artifacts/runs/my-run
uv run agent-eval score tasks/click/click-3449/task.yaml \
  artifacts/runs/my-run/patch.diff
```

Each attempt directory contains `patch.diff`, `agent-output.txt`,
`metadata.json`, and `trace.jsonl`. The held-out tests are only installed in the
fresh scorer workspace; the agent workspace is destroyed before scoring starts.

Events are appended to `artifacts/evaluations.jsonl` by default. Use
`--trace-file PATH` to choose another destination. Cost is recorded when the
agent provider reports it; the current Cursor SDK does not expose monetary cost.

Each task may declare a `setup_command` and `setup_timeout_seconds`. Setup runs
inside the disposable checkout before the agent starts, so repository
dependencies are isolated from the harness environment. The Click task uses
`uv sync` to create its own `.venv`, then scores with `uv run pytest`.

## Optional failure judge

Deterministic scoring never calls an LLM. After a failed score, you can ask an
OpenAI model to classify the failure mode. Pass/fail is still the held-out
tests; the judge only writes `judge.json` and a `judge_completed` trace event.

```bash
export OPENAI_API_KEY="sk-..."
uv run agent-eval run tasks/click/click-3449/task.yaml \
  --model "your-model-id" --judge
uv run agent-eval judge tasks/click/click-3449/task.yaml artifacts/runs/my-run
```

`score` remains unpaid and model-free. `--judge` is off by default and requires
`OPENAI_API_KEY` before a paid coding-agent attempt starts.
