from pathlib import Path
from typing import Annotated, NoReturn

import typer

from .agent import AgentExecutionError, CursorCodingAgent
from .artifacts import ArtifactError, RunArtifactStore
from .models import EvaluationStatus
from .runner import (
    AttemptRunner,
    EvaluationRunner,
    EvaluationRunnerError,
    InvalidPatchError,
    PatchScorer,
)
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
def attempt(
    task_file: Path,
    model: Annotated[
        str,
        typer.Option(help="Cursor model ID used for the attempt."),
    ],
    output: Annotated[
        Path | None,
        typer.Option(help="Exact directory for persisted run artifacts."),
    ] = None,
) -> None:
    """Run and persist one paid agent attempt without scoring it."""
    task = load_task(task_file)
    runner = AttemptRunner(
        agent=CursorCodingAgent(model),
        sandbox=LocalSandbox(),
        artifact_store=RunArtifactStore(),
    )
    try:
        result = runner.create_attempt(task, output)
    except (AgentExecutionError, ArtifactError, EvaluationRunnerError, SandboxError) as error:
        _exit_with_error(error, _error_status(error))

    typer.echo(f"ATTEMPT: {result.run_id} in {result.duration_seconds:.2f}s")
    typer.echo(f"Agent status: {result.agent_status}")
    typer.echo(f"Patch: {result.patch_path}")
    typer.echo(f"Artifacts: {result.artifact_dir}")
    if result.usage is not None:
        typer.echo(f"Tokens: {result.usage.total_tokens}")
    if result.agent_status != "finished":
        raise typer.Exit(code=1)


@app.command()
def score(task_file: Path, patch_file: Path) -> None:
    """Score a persisted patch in a fresh checkout without an agent call."""
    task = load_task(task_file)
    tracer = JsonlTracer(patch_file.parent / "trace.jsonl")
    try:
        result = PatchScorer(LocalSandbox()).score_patch(
            task,
            patch_file,
            tracer,
        )
    except (InvalidPatchError, EvaluationRunnerError, SandboxError) as error:
        _exit_with_error(error, _error_status(error))

    typer.echo(
        f"{result.status.value.upper()}: {task.id} "
        f"in {result.duration_seconds:.2f}s"
    )
    typer.echo(f"Environment: {result.environment_hash}")
    if not result.passed:
        raise typer.Exit(code=1)


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
    output: Annotated[
        Path | None,
        typer.Option(help="Exact directory for persisted run artifacts."),
    ] = None,
) -> None:
    """Create an attempt, then score its patch in a fresh checkout."""
    task = load_task(task_file)
    runner = EvaluationRunner(
        agent=CursorCodingAgent(model),
        agent_sandbox=LocalSandbox(),
        scorer_sandbox=LocalSandbox(),
        artifact_store=RunArtifactStore(),
        tracer=JsonlTracer(trace_file),
    )

    try:
        result = runner.run(task, output)
    except (
        AgentExecutionError,
        ArtifactError,
        EvaluationRunnerError,
        SandboxError,
    ) as error:
        _exit_with_error(error, _error_status(error))

    outcome = "PASS" if result.passed else "FAIL"
    typer.echo(
        f"{outcome}: {task.id} "
        f"(attempt {result.attempt.duration_seconds:.2f}s, "
        f"score {result.score.duration_seconds:.2f}s)"
    )
    typer.echo(f"Status: {result.status.value}")
    typer.echo(f"Patch: {result.attempt.patch_path}")
    if result.attempt.usage is not None:
        typer.echo(f"Tokens: {result.attempt.usage.total_tokens}")
    if result.attempt.cost_usd is not None:
        typer.echo(f"Cost: ${result.attempt.cost_usd:.4f}")
    typer.echo(f"Trace: {trace_file}")

    if not result.passed:
        raise typer.Exit(code=1)


def _exit_with_error(
    error: Exception,
    status: EvaluationStatus,
) -> NoReturn:
    typer.echo(f"ERROR [{status.value}]: {error}", err=True)
    raise typer.Exit(code=1) from error


def _error_status(error: Exception) -> EvaluationStatus:
    if isinstance(error, AgentExecutionError):
        return EvaluationStatus.AGENT_ERROR
    return getattr(error, "status", EvaluationStatus.SCORER_ERROR)
