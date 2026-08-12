from pathlib import Path

import typer

from .task_loader import load_task

app = typer.Typer(no_args_is_help=True)


@app.command()
def inspect(task_file: Path) -> None:
    """Validate and summarize a task manifest."""
    task = load_task(task_file)
    typer.echo(f"{task.id}: {task.repository}@{task.base_commit[:12]}")


@app.command()
def run(task_file: Path, model: str = typer.Option(...)) -> None:
    """Run one evaluation attempt."""
    task = load_task(task_file)
    typer.echo(f"Runner not implemented yet: {task.id} with {model}")
