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
uv run pytest
uv run agent-eval inspect tasks/click/click-3449/task.yaml
```

## First milestone

Implement one complete command that checks out the pinned Click commit in an
isolated container, lets an agent attempt the issue, mounts the held-out test,
and records the patch, test result, duration, token usage, and cost:

```bash
uv run agent-eval run tasks/click/click-3449/task.yaml --model <model>
```

The held-out tests are public benchmark assets, but they must not be mounted in
the agent's workspace until its attempt is complete.
