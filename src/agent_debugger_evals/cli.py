from pathlib import Path
from typing import Annotated

import typer

from .agent import AgentExecutionError, CursorCodingAgent
from .runner import EvaluationRunner, EvaluationRunnerError
from .sandbox import LocalSandbox, SandboxError
from .task_loader import load_task
from .tracing import JsonlTracer

app = typer.Typer(no_args_is_help=True)


@app.command()
def inspect(task_file: Path) -> None:
    """Validate and summarize a task manifest."""
    task = load_task(task_file)
    typer.echo(f"{task.id}: {task.repository}@{task.base_commit[:12]}")


@app.command()
def run(
    task_file: Path,
    model: Annotated[
        str,
        typer.Option(help="Cursor model ID used for the attempt."),
    ],
    trace_file: Annotated[
        Path,
        typer.Option(help="JSONL file to append evaluation events to."),
    ] = Path("artifacts/evaluations.jsonl"),
) -> None:
    """Run one evaluation attempt."""
    task = load_task(task_file)
    runner = EvaluationRunner(
        agent=CursorCodingAgent(model),
        sandbox=LocalSandbox(),
        tracer=JsonlTracer(trace_file),
    )

    try:
        result = runner.run(task)
    except (AgentExecutionError, EvaluationRunnerError, SandboxError) as error:
        typer.echo(f"ERROR: {error}", err=True)
        raise typer.Exit(code=1) from error

    outcome = "PASS" if result.passed else "FAIL"
    typer.echo(f"{outcome}: {task.id} in {result.duration_seconds:.2f}s")
    if result.usage is not None:
        typer.echo(f"Tokens: {result.usage.total_tokens}")
    if result.cost_usd is not None:
        typer.echo(f"Cost: ${result.cost_usd:.4f}")
    typer.echo(f"Trace: {trace_file}")

    if not result.passed:
        raise typer.Exit(code=1)
