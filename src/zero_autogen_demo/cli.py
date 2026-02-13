from __future__ import annotations

import asyncio
import os
from pathlib import Path
from urllib.parse import urlparse

import typer
from dotenv import load_dotenv

from .agent import audit_run, replay_run, run_autonomous_demo, show_tidb_zero_connection
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


def is_timeout_like_error(exc: Exception) -> bool:
    text = str(exc).lower()
    name = type(exc).__name__.lower()
    timeout_markers = [
        "timeout",
        "timed out",
        "readtimeout",
        "apitimeouterror",
    ]
    return any(marker in text for marker in timeout_markers) or any(marker in name for marker in timeout_markers)


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
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="Model provider override: openai, anthropic, gemini, or openai_compatible.",
    ),
    model: str | None = typer.Option(None, "--model", help="Override model name (default from MODEL_NAME)."),
    model_base_url: str | None = typer.Option(
        None,
        "--model-base-url",
        help="Override model API base URL (useful for OpenAI-compatible endpoints).",
    ),
    model_organization: str | None = typer.Option(
        None,
        "--model-organization",
        help="Optional model organization header (used by some providers).",
    ),
    tag: str | None = typer.Option(None, "--tag", help="Override TIDB_ZERO_TAG for provisioning traceability."),
    model_timeout_sec: int | None = typer.Option(
        None,
        "--model-timeout-sec",
        help="Override model API timeout in seconds.",
    ),
    model_max_retries: int | None = typer.Option(
        None,
        "--model-max-retries",
        help="Override model API max retries.",
    ),
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
            model_provider=provider,
            model_name=model,
            model_base_url=model_base_url,
            model_organization=model_organization,
            tidb_zero_tag=tag,
            model_timeout_sec=model_timeout_sec,
            model_max_retries=model_max_retries,
            max_tool_iterations=max_tool_iterations,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    try:
        result = asyncio.run(run_autonomous_demo(settings, resolved_goal, resolved_source_url))
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"[ERROR] {exc}")
        if is_timeout_like_error(exc):
            typer.echo(
                "[HINT] Model request timed out. Increase MODEL_TIMEOUT_SEC (for example 120 or 180), "
                "or pass --model-timeout-sec 180 and retry."
            )
        raise typer.Exit(code=1)
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


@app.command("conn")
def conn_command(
    run_id: str = typer.Argument(..., help="Run ID from a previous `run` command."),
    redact_password: bool = typer.Option(
        False,
        "--redact-password",
        help="Hide the password in output.",
    ),
) -> None:
    """Show TiDB Zero connection details for a run (includes DB user/password by default)."""
    load_dotenv()
    try:
        show_tidb_zero_connection(resolve_runs_dir(), run_id, redact_password=redact_password)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


if __name__ == "__main__":
    app()
