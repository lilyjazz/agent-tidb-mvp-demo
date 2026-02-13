from __future__ import annotations

import asyncio
import os
from pathlib import Path
from urllib.parse import urlparse

import typer
from dotenv import load_dotenv

from .agent import audit_run, replay_run, run_autonomous_demo
from .config import Settings


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "AutoGen + TiDB Zero CLI demo. "
        "Shows full agent flow with TiDB Zero provisioning, autonomous schema design, and autonomous SQL."
    ),
)


def resolve_runs_dir() -> Path:
    return Path(os.getenv("RUNS_DIR", ".runs"))


def resolve_non_empty(value: str | None, prompt_text: str) -> str:
    if value is not None and value.strip():
        return value.strip()

    while True:
        entered = typer.prompt(prompt_text).strip()
        if entered:
            return entered
        typer.echo("Input cannot be empty.")


def validate_source_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise typer.BadParameter("source-url must be a valid http or https URL.")
    return url


@app.command("run")
def run_command(
    goal: str | None = typer.Argument(
        None,
        help="High-level objective for the autonomous agent. If omitted, CLI will prompt for it.",
    ),
    source_url: str | None = typer.Option(
        None,
        "--source-url",
        help="Public data source URL the agent should use. If omitted, CLI will prompt for it.",
    ),
    model: str | None = typer.Option(None, "--model", help="Override model name (default from MODEL_NAME)."),
    tag: str | None = typer.Option(None, "--tag", help="Override TIDB_ZERO_TAG for provisioning traceability."),
    max_tool_iterations: int | None = typer.Option(
        None,
        "--max-tool-iterations",
        help="Override maximum tool-call iterations for AutoGen AssistantAgent.",
    ),
) -> None:
    """Provision TiDB Zero and execute a full autonomous run."""
    load_dotenv()

    resolved_goal = resolve_non_empty(goal, "Enter your goal/question")
    resolved_source_url = validate_source_url(resolve_non_empty(source_url, "Enter source URL"))

    try:
        settings = Settings.from_env(
            model_name=model,
            tidb_zero_tag=tag,
            max_tool_iterations=max_tool_iterations,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    result = asyncio.run(run_autonomous_demo(settings, resolved_goal, resolved_source_url))
    print(f"[RUN_COMPLETE] run_id={result.run_id} run_dir={result.run_dir}")


@app.command("replay")
def replay_command(
    run_id: str = typer.Argument(..., help="Run ID from a previous `run` command."),
) -> None:
    """Replay the human-readable timeline of a previous run."""
    load_dotenv()
    try:
        replay_run(resolve_runs_dir(), run_id)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("audit")
def audit_command(
    run_id: str = typer.Argument(..., help="Run ID from a previous `run` command."),
) -> None:
    """Print SQL audit trail and autonomy proof for a run."""
    load_dotenv()
    try:
        audit_run(resolve_runs_dir(), run_id)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc


if __name__ == "__main__":
    app()
