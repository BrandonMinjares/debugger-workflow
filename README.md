# Agent Debugger Evals

A reproducible evaluation and debugging harness for coding agents. The first
benchmark task recreates [Click issue #3449](https://github.com/pallets/click/issues/3449)
from a pinned pre-fix commit and scores the resulting patch with held-out tests.

## Structure

```text
tasks/                       Curated repository tasks and held-out tests
src/agent_debugger_evals/
  agent.py                   Provider-neutral coding-agent interface
  sandbox.py                 Isolated checkout interface
  runner.py                  Evaluation orchestration
  scoring.py                 Deterministic test scoring
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

## First milestone

The run command checks out the pinned Click commit in a disposable workspace,
lets a Cursor agent attempt the issue, mounts the held-out test afterward, and
records the patch, test result, duration, and token usage as JSONL:

```bash
uv run agent-eval run tasks/click/click-3449/task.yaml --model "your-model-id"
```

The held-out tests are public benchmark assets, but they must not be mounted in
the agent's workspace until its attempt is complete.

Events are appended to `artifacts/evaluations.jsonl` by default. Use
`--trace-file PATH` to choose another destination. Cost is recorded when the
agent provider reports it; the current Cursor SDK does not expose monetary cost.

Each task may declare a `setup_command` and `setup_timeout_seconds`. Setup runs
inside the disposable checkout before the agent starts, so repository
dependencies are isolated from the harness environment. The Click task uses
`uv sync` to create its own `.venv`, then scores with `uv run pytest`.
