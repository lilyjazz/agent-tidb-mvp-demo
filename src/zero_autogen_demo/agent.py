from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from autogen_agentchat.agents import AssistantAgent

from .config import Settings
from .db import TiDBSandbox, json_dumps
from .zero import ZeroInstance, provision_zero_instance


CREATE_TABLE_RE = re.compile(
    r"^\s*CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?`?([a-zA-Z0-9_.]+)`?",
    flags=re.IGNORECASE,
)

SUBSCRIPTION_PROVIDERS = {"claude_subscription", "codex_subscription"}
MAX_BATCH_TOOL_CALLS = 3


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_ms(ms: int) -> str:
    return f"{(max(ms, 0) / 1000):.3f}s"


def format_duration_map(ms_by_key: dict[str, int]) -> str:
    if not ms_by_key:
        return ""

    parts: list[str] = []
    for key, value in sorted(ms_by_key.items(), key=lambda item: (-item[1], item[0])):
        safe_value = max(int(value), 0)
        parts.append(f"{key}={safe_value}ms ({format_ms(safe_value)})")
    return " ".join(parts)


def truncate_text(text: str, limit: int = 320) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 15] + "...[truncated]"


def truncate_middle_text(text: str, limit: int = 320) -> str:
    if len(text) <= limit:
        return text
    marker = "...[truncated]..."
    budget = limit - len(marker)
    if budget <= 20:
        return text[: max(limit - 3, 0)] + "..."
    head = budget // 2
    tail = budget - head
    return text[:head] + marker + text[-tail:]


def normalize_sql(raw_sql: str) -> str:
    sql = raw_sql.strip()
    if sql.startswith("```") and sql.endswith("```"):
        lines = sql.splitlines()
        if len(lines) >= 3:
            sql = "\n".join(lines[1:-1]).strip()
    return sql


def extract_json_argument(arguments: str, key: str) -> str | None:
    parsed = safe_json_loads(arguments)
    if isinstance(parsed, dict):
        value = parsed.get(key)
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
    return None


def classify_sql(sql: str) -> str:
    first = sql.lstrip().split(None, 1)
    token = first[0].upper() if first else ""
    if token in {"CREATE", "ALTER", "DROP", "TRUNCATE", "RENAME"}:
        return "DDL"
    if token in {"INSERT", "UPDATE", "DELETE", "REPLACE", "MERGE"}:
        return "DML"
    if token in {"SELECT", "WITH", "SHOW", "DESCRIBE", "DESC", "EXPLAIN"}:
        return "QUERY"
    return "OTHER"


def extract_created_table(sql: str) -> str | None:
    match = CREATE_TABLE_RE.match(sql)
    if not match:
        return None
    return match.group(1).split(".")[-1].strip("`")


def safe_json_loads(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def to_serializable(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): to_serializable(v, depth=depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item, depth=depth + 1) for item in value]
    if hasattr(value, "model_dump"):
        return to_serializable(value.model_dump(), depth=depth + 1)
    if hasattr(value, "__dict__"):
        raw = {
            k: v
            for k, v in vars(value).items()
            if not k.startswith("_")
        }
        return to_serializable(raw, depth=depth + 1)
    return str(value)


def summarize_event(event: Any) -> dict[str, Any]:
    return {
        "source": getattr(event, "source", None),
        "content": to_serializable(getattr(event, "content", None)),
        "metadata": to_serializable(getattr(event, "metadata", None)),
    }


@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    run_dir: Path
    timeline_path: Path
    events_path: Path
    sql_audit_path: Path
    step_perf_path: Path
    perf_path: Path
    instance_path: Path
    final_path: Path
    error_trace_path: Path


@dataclass
class RunTracker:
    artifacts: RunArtifacts
    sql_count: int = 0
    ddl_count: int = 0
    created_tables: set[str] = field(default_factory=set)
    final_answer: str = ""
    last_assistant_text: str = ""
    step_count: int = 0
    model_decision_ms: int = 0

    def emit(self, kind: str, message: str) -> None:
        ts = utc_now_iso()
        line = f"[{ts}][{kind}] {message}"
        print(line, flush=True)
        self._append_jsonl(
            self.artifacts.timeline_path,
            {
                "ts": ts,
                "kind": kind,
                "message": message,
            },
        )

    def append_raw_event(self, step_no: int, event_type: str, payload: dict[str, Any]) -> None:
        self._append_jsonl(
            self.artifacts.events_path,
            {
                "ts": utc_now_iso(),
                "step_no": step_no,
                "event_type": event_type,
                "payload": payload,
            },
        )

    def append_sql_audit(self, record: dict[str, Any]) -> None:
        self._append_jsonl(self.artifacts.sql_audit_path, record)

    def append_step_perf(self, record: dict[str, Any]) -> None:
        self._append_jsonl(self.artifacts.step_perf_path, record)

    def record_sql(self, statement_type: str, sql: str, *, is_error: bool) -> None:
        self.sql_count += 1
        if statement_type == "DDL" and not is_error:
            self.ddl_count += 1
            table_name = extract_created_table(sql)
            if table_name:
                self.created_tables.add(table_name)

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json_dumps(payload))
            f.write("\n")


class ToolRuntime:
    def __init__(
        self,
        *,
        sandbox: TiDBSandbox,
        tracker: RunTracker,
        run_id: str,
        sql_row_limit: int,
        fetch_max_chars: int,
        http_timeout_sec: int,
    ) -> None:
        self.sandbox = sandbox
        self.tracker = tracker
        self.run_id = run_id
        self.sql_row_limit = sql_row_limit
        self.fetch_max_chars = fetch_max_chars
        self.http_timeout_sec = http_timeout_sec
        self.total_tool_exec_ms = 0
        self.total_db_exec_ms = 0
        self.tool_exec_ms_by_name: dict[str, int] = {}
        self.db_exec_ms_by_statement_type: dict[str, int] = {}
        self.db_exec_ms_by_task: dict[str, int] = {}

    def _record_tool_elapsed(self, started_at: float, *, tool_name: str | None = None) -> int:
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        safe_elapsed_ms = max(elapsed_ms, 0)
        self.total_tool_exec_ms += safe_elapsed_ms
        if tool_name:
            self.tool_exec_ms_by_name[tool_name] = self.tool_exec_ms_by_name.get(tool_name, 0) + safe_elapsed_ms
        return safe_elapsed_ms

    async def http_fetch(self, url: str) -> str:
        """Fetch a public HTTP/HTTPS URL with GET and return status, headers, and body text."""
        started_at = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.http_timeout_sec, follow_redirects=True) as client:
                response = await client.get(url)

            body = response.text
            if len(body) > self.fetch_max_chars:
                body = body[: self.fetch_max_chars] + "\n...[TRUNCATED]"

            payload = {
                "url": str(response.url),
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type"),
                "body": body,
            }
            return json_dumps(payload)
        finally:
            self._record_tool_elapsed(started_at, tool_name="http_fetch")

    def log_thought(self, thought: str) -> str:
        """Record a concise thought summary before taking an action."""
        started_at = time.perf_counter()
        cleaned = thought.strip()
        if not cleaned:
            self._record_tool_elapsed(started_at, tool_name="log_thought")
            return json_dumps({"ok": False, "error": "thought cannot be empty"})
        payload = json_dumps({"ok": True, "thought": truncate_text(cleaned, 800)})
        self._record_tool_elapsed(started_at, tool_name="log_thought")
        return payload

    def schema_inspect(self) -> str:
        """Inspect current database schema and return tables/columns metadata as JSON."""
        started_at = time.perf_counter()
        try:
            sql = """
            SELECT
              table_name,
              column_name,
              data_type,
              column_type,
              is_nullable,
              column_key
            FROM information_schema.columns
            WHERE table_schema = DATABASE()
            ORDER BY table_name, ordinal_position
            """
            result, elapsed_ms = self.sandbox.execute_sql(sql, max_rows=2000)
            safe_db_ms = max(elapsed_ms, 0)
            self.total_db_exec_ms += safe_db_ms
            self.db_exec_ms_by_task["schema_inspect"] = self.db_exec_ms_by_task.get("schema_inspect", 0) + safe_db_ms
            return json_dumps(result)
        finally:
            self._record_tool_elapsed(started_at, tool_name="schema_inspect")

    def sql_exec(self, sql: str) -> str:
        """Execute exactly one SQL statement against TiDB Zero and return JSON result."""
        started_at = time.perf_counter()
        try:
            normalized_sql = normalize_sql(sql)
            statement_type = classify_sql(normalized_sql)
            elapsed_ms = 0
            result_rows: int | None = None
            error_text: str | None = None
            db_started_at = time.perf_counter()

            try:
                result, elapsed_ms = self.sandbox.execute_sql(normalized_sql, max_rows=self.sql_row_limit)
                if result.get("type") == "query":
                    result_rows = int(result.get("row_count", 0))
                else:
                    affected = result.get("affected_rows")
                    result_rows = int(affected) if isinstance(affected, int) else None
                ok = True
            except Exception as exc:  # noqa: BLE001
                elapsed_ms = int((time.perf_counter() - db_started_at) * 1000)
                result = {
                    "type": "error",
                    "error": str(exc),
                }
                error_text = str(exc)
                ok = False

            safe_db_ms = max(elapsed_ms, 0)
            self.total_db_exec_ms += safe_db_ms
            self.db_exec_ms_by_task["sql_exec"] = self.db_exec_ms_by_task.get("sql_exec", 0) + safe_db_ms
            self.db_exec_ms_by_statement_type[statement_type] = (
                self.db_exec_ms_by_statement_type.get(statement_type, 0) + safe_db_ms
            )

            try:
                self.sandbox.log_sql_audit(
                    run_id=self.run_id,
                    statement_type=statement_type,
                    sql_text=normalized_sql,
                    is_error=not ok,
                    result_rows=result_rows,
                    elapsed_ms=elapsed_ms,
                    error_text=error_text,
                )
            except Exception as log_exc:  # noqa: BLE001
                suffix = f"sql_audit_log_error: {log_exc}"
                error_text = f"{error_text} | {suffix}" if error_text else suffix

            audit_record = {
                "ts": utc_now_iso(),
                "statement_type": statement_type,
                "sql_text": normalized_sql,
                "is_error": not ok,
                "result_rows": result_rows,
                "elapsed_ms": elapsed_ms,
                "error_text": error_text,
            }
            self.tracker.append_sql_audit(audit_record)
            self.tracker.record_sql(statement_type, normalized_sql, is_error=not ok)

            response = {
                "ok": ok,
                "statement_type": statement_type,
                "elapsed_ms": elapsed_ms,
                "result": result,
                "error": error_text,
            }
            return json_dumps(response)
        finally:
            self._record_tool_elapsed(started_at, tool_name="sql_exec")


def build_system_message(database_name: str) -> str:
    return (
        "You are an autonomous data agent operating inside a dedicated TiDB Zero database. "
        f"Your active schema is `{database_name}`. "
        "You have full SQL privileges in this schema. "
        "CRITICAL: Data fetched via http_fetch is NOT automatically stored in the database. "
        "You MUST explicitly `CREATE TABLE` and `INSERT` the fetched data before you can query it with SQL. "
        "Do not look for data in system tables like `run_logs` or `step_logs`. "
        "Do not assume predefined business tables. Design your own schema based on observed data. "
        "Use tools to fetch data, inspect schema, and execute SQL. "
        "Before each meaningful action, call log_thought with a concise 1-2 sentence rationale. "
        "Use concise thought summaries only, not hidden chain-of-thought. "
        "When using sql_exec, execute exactly one SQL statement per call. "
        "Show robust behavior: inspect data first, create schema, ingest representative records, then analyze. "
        "Your final response must include: (1) schema rationale, (2) key SQL evidence, (3) answer. "
        "End final response with the keyword TERMINATE."
    )


def build_task(goal: str, source_url: str) -> str:
    return (
        "Run an end-to-end autonomous data workflow.\n"
        f"Goal: {goal}\n"
        f"Public source URL: {source_url}\n"
        "Constraints:\n"
        "- Fetch and inspect source data first.\n"
        "- Design schema yourself (no preset schema).\n"
        "- Create table(s), ingest representative rows, and run analytical SQL.\n"
        "- Keep SQL auditable and deterministic.\n"
        "- Include SQL evidence in final answer."
    )


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


def is_subscription_provider(provider: str) -> bool:
    return provider in SUBSCRIPTION_PROVIDERS


def ensure_subscription_cli_available(settings: Settings) -> None:
    if settings.model_provider == "codex_subscription":
        if shutil.which(settings.codex_subscription_bin) is None:
            raise RuntimeError(
                f"Codex subscription mode requires '{settings.codex_subscription_bin}' in PATH. "
                "Install Codex CLI and run `codex login` first."
            )
    if settings.model_provider == "claude_subscription":
        if shutil.which(settings.claude_subscription_bin) is None:
            raise RuntimeError(
                f"Claude subscription mode requires '{settings.claude_subscription_bin}' in PATH. "
                "Install Claude Code CLI and run `claude` (or `/login`) first."
            )


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    direct = safe_json_loads(text)
    if isinstance(direct, dict):
        return direct

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    sliced = text[start : end + 1]
    parsed = safe_json_loads(sliced)
    if isinstance(parsed, dict):
        return parsed
    return None


def build_subscription_step_prompt(
    *,
    goal: str,
    source_url: str,
    history: list[dict[str, Any]],
    database_name: str,
    batch_tools_enabled: bool,
) -> str:
    history_window = history[-8:]
    history_json = json_dumps(history_window)

    if batch_tools_enabled:
        output_schema_hint = (
            "Return exactly one JSON object with this schema and no markdown/code fences:\n"
            '{"thought":"short rationale","tool_calls":[{"name":"http_fetch|schema_inspect|sql_exec","args":{}}],"final_answer":null}\n'
            "or\n"
            '{"thought":"short rationale","tool_call":{"name":"http_fetch|schema_inspect|sql_exec","args":{}},"final_answer":null}\n'
            "or\n"
            '{"thought":"short rationale","tool_call":null,"final_answer":"final answer with SQL evidence"}\n\n'
        )
        rules_hint = (
            "Rules:\n"
            "- Keep `thought` to one or two short sentences.\n"
            f"- Prefer `tool_calls` for deterministic multi-step sequences (2-{MAX_BATCH_TOOL_CALLS} actions), executed in order.\n"
            "- Use `tool_call` for one-off or uncertain actions.\n"
            "- Decision policy: if source data is not fetched yet, do one `http_fetch`; once fetched, batch the remaining deterministic SQL steps.\n"
            "- After any tool error in the most recent step, use single-action corrective `tool_call` first.\n"
            "- Tool args contract: http_fetch -> {\"url\":\"...\"}; schema_inspect -> {}; sql_exec -> {\"sql\":\"...\"}.\n"
            "- `sql_exec` must contain exactly one SQL statement in args.sql.\n"
            "- For loading fetched data, prefer explicit `CREATE TABLE IF NOT EXISTS` + `INSERT` + `SELECT`; avoid `CREATE TABLE ... SELECT`.\n"
            "- If task is complete, set final_answer and tool_call=null.\n"
            "- Do not include any keys outside the required schema.\n\n"
        )
    else:
        output_schema_hint = (
            "Return exactly one JSON object with this schema and no markdown/code fences:\n"
            '{"thought":"short rationale","tool_call":{"name":"http_fetch|schema_inspect|sql_exec","args":{}},"final_answer":null}\n'
            "or\n"
            '{"thought":"short rationale","tool_call":null,"final_answer":"final answer with SQL evidence"}\n\n'
        )
        rules_hint = (
            "Rules:\n"
            "- Keep `thought` to one or two short sentences.\n"
            "- Use only one tool call per step.\n"
            "- Tool args contract: http_fetch -> {\"url\":\"...\"}; schema_inspect -> {}; sql_exec -> {\"sql\":\"...\"}.\n"
            "- `sql_exec` must contain exactly one SQL statement in args.sql.\n"
            "- If task is complete, set final_answer and tool_call=null.\n"
            "- Do not include any keys outside the required schema.\n\n"
        )

    return (
        "You are controlling a tool-using data agent loop.\n"
        f"{output_schema_hint}"
        f"{rules_hint}"
        "CRITICAL: Data fetched via http_fetch is NOT automatically stored in the database.\n"
        "You MUST explicitly `CREATE TABLE` and `INSERT` the fetched data before you can query it with SQL.\n"
        "Avoid `CREATE TABLE ... SELECT`; use explicit `CREATE TABLE` then `INSERT`.\n"
        "Do NOT look for data in system tables like `run_logs` or `step_logs`.\n\n"
        f"Task goal: {goal}\n"
        f"Public source URL: {source_url}\n"
        f"Active database schema: {database_name}\n"
        "Recent steps JSON:\n"
        f"{history_json}\n"
    )


def subscription_tool_call_schema() -> dict[str, Any]:
    return {
        "anyOf": [
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "args"],
                "properties": {
                    "name": {"type": "string", "const": "http_fetch"},
                    "args": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["url"],
                        "properties": {
                            "url": {"type": "string"},
                        },
                    },
                },
            },
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "args"],
                "properties": {
                    "name": {"type": "string", "const": "schema_inspect"},
                    "args": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [],
                        "properties": {},
                    },
                },
            },
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "args"],
                "properties": {
                    "name": {"type": "string", "const": "sql_exec"},
                    "args": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["sql"],
                        "properties": {
                            "sql": {"type": "string"},
                        },
                    },
                },
            },
        ]
    }


def subscription_decision_schema(*, batch_tools_enabled: bool) -> dict[str, Any]:
    tool_call_schema = subscription_tool_call_schema()
    base_single = {
        "type": "object",
        "additionalProperties": False,
        "required": ["thought", "tool_call", "final_answer"],
        "properties": {
            "thought": {"type": "string"},
            "tool_call": {
                "anyOf": [
                    {"type": "null"},
                    tool_call_schema,
                ]
            },
            "final_answer": {"type": ["string", "null"]},
        },
    }

    if not batch_tools_enabled:
        return base_single

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["thought", "final_answer"],
        "properties": {
            "thought": {"type": "string"},
            "tool_call": {
                "anyOf": [
                    {"type": "null"},
                    tool_call_schema,
                ]
            },
            "tool_calls": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_BATCH_TOOL_CALLS,
                        "items": tool_call_schema,
                    },
                ]
            },
            "final_answer": {"type": ["string", "null"]},
        },
    }


def run_codex_subscription_prompt(settings: Settings, prompt: str, *, batch_tools_enabled: bool) -> str:
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        output_path = Path(tmp.name)

    schema_path: Path | None = None
    if not batch_tools_enabled:
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tmp_schema:
            schema_path = Path(tmp_schema.name)
            tmp_schema.write(json.dumps(subscription_decision_schema(batch_tools_enabled=False)))

    cmd = [
        settings.codex_subscription_bin,
        "exec",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--output-last-message",
        str(output_path),
    ]

    if schema_path is not None:
        cmd.extend(["--output-schema", str(schema_path)])

    # Only add --model if it is not "default" or empty
    # Codex CLI with ChatGPT login often fails if --model is specified explicitly
    if settings.model_name and settings.model_name.lower() != "default":
        cmd.extend(["--model", settings.model_name])

    cmd.append(prompt)

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=settings.model_timeout_sec,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Codex CLI binary not found: {settings.codex_subscription_bin}. Install Codex CLI first."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Codex CLI timed out after {settings.model_timeout_sec}s") from exc

    try:
        output_text = output_path.read_text(encoding="utf-8").strip() if output_path.exists() else ""
    finally:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        if schema_path is not None and schema_path.exists():
            schema_path.unlink(missing_ok=True)

    if not output_text:
        output_text = (completed.stdout or "").strip()

    if not output_text:
        stderr_json = extract_first_json_object(completed.stderr or "")
        if isinstance(stderr_json, dict):
            output_text = json_dumps(stderr_json)

    if completed.returncode != 0 and not output_text:
        stderr = truncate_middle_text((completed.stderr or "").strip(), 1800)
        raise RuntimeError(f"Codex CLI failed with code {completed.returncode}: {stderr}")

    if not output_text:
        raise RuntimeError("Codex CLI returned empty output")

    return output_text


def run_claude_subscription_prompt(settings: Settings, prompt: str, *, batch_tools_enabled: bool) -> str:
    cmd = [
        settings.claude_subscription_bin,
        "-p",
        "--output-format",
        "text",
        "--json-schema",
        json.dumps(subscription_decision_schema(batch_tools_enabled=batch_tools_enabled)),
        "--max-turns",
        "1",
        "--model",
        settings.model_name,
        "--tools",
        "",
        prompt,
    ]

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=settings.model_timeout_sec,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Claude CLI binary not found: {settings.claude_subscription_bin}. Install Claude Code CLI first."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Claude CLI timed out after {settings.model_timeout_sec}s") from exc

    output_text = (completed.stdout or "").strip()
    if completed.returncode != 0 and not output_text:
        stderr = truncate_text((completed.stderr or "").strip(), 500)
        raise RuntimeError(f"Claude CLI failed with code {completed.returncode}: {stderr}")

    if not output_text:
        raise RuntimeError("Claude CLI returned empty output")

    return output_text


def is_retryable_subscription_error(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = [
        "timeout",
        "timed out",
        "network error",
        "transport error",
        "stream disconnected",
        "connection reset",
        "temporary failure",
        "temporarily unavailable",
        "rate limit",
        "429",
    ]
    return any(marker in text for marker in markers)


async def request_subscription_decision(settings: Settings, prompt: str, *, batch_tools_enabled: bool) -> str:
    if settings.model_provider == "codex_subscription":
        runner = run_codex_subscription_prompt
    elif settings.model_provider == "claude_subscription":
        runner = run_claude_subscription_prompt
    else:
        raise RuntimeError(f"Unsupported subscription provider: {settings.model_provider}")

    max_attempts = max(settings.model_max_retries, 0) + 1
    delay_sec = 1.0
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await asyncio.to_thread(
                runner,
                settings,
                prompt,
                batch_tools_enabled=batch_tools_enabled,
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            should_retry = is_retryable_subscription_error(exc) and attempt < max_attempts
            if not should_retry:
                raise
            await asyncio.sleep(delay_sec)
            delay_sec = min(delay_sec * 2, 4.0)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Subscription decision failed without a specific error")


def summarize_observation_for_history(text: str, limit: int = 1800) -> str:
    return truncate_text(text.replace("\n", " "), limit)


def emit_sql_observation(observation: str, tracker: RunTracker) -> None:
    parsed = safe_json_loads(observation)
    if isinstance(parsed, dict):
        statement_type = str(parsed.get("statement_type", "OTHER"))
        ok = bool(parsed.get("ok"))
        raw_result = parsed.get("result")
        result_payload: dict[str, Any] = raw_result if isinstance(raw_result, dict) else {}
        if ok and result_payload.get("type") == "query":
            rows = result_payload.get("row_count")
            tracker.emit("OBSERVATION", f"sql_exec {statement_type} ok rows={rows}")
            return
        if ok:
            affected = result_payload.get("affected_rows")
            tracker.emit("OBSERVATION", f"sql_exec {statement_type} ok affected_rows={affected}")
            return
        error_text = str(parsed.get("error") or observation)
        tracker.emit("OBSERVATION", f"sql_exec {statement_type} error={truncate_text(error_text, 360)}")
        return

    tracker.emit("OBSERVATION", f"sql_exec raw={truncate_text(observation, 360)}")


def subscription_observation_is_error(observation: str) -> bool:
    parsed = safe_json_loads(observation)
    if isinstance(parsed, dict) and "ok" in parsed:
        return not bool(parsed.get("ok"))
    return False


def parse_subscription_tool_calls(
    parsed: dict[str, Any],
    *,
    batch_tools_enabled: bool,
) -> tuple[list[dict[str, Any]], bool, str | None]:
    if batch_tools_enabled:
        raw_batch = parsed.get("tool_calls")
        if raw_batch is not None:
            if not isinstance(raw_batch, list):
                return [], True, "tool_calls must be an array when provided."
            if len(raw_batch) == 0:
                return [], True, "tool_calls cannot be empty."
            if len(raw_batch) > MAX_BATCH_TOOL_CALLS:
                return (
                    [],
                    True,
                    f"tool_calls exceeds max size {MAX_BATCH_TOOL_CALLS}; use fewer actions.",
                )

            tool_calls: list[dict[str, Any]] = []
            for index, item in enumerate(raw_batch, start=1):
                if not isinstance(item, dict):
                    return [], True, f"tool_calls[{index}] must be an object."
                tool_calls.append(item)
            return tool_calls, True, None

    raw_single = parsed.get("tool_call")
    if isinstance(raw_single, dict):
        return [raw_single], False, None

    return [], False, "Missing tool_call and no final_answer provided."


async def execute_subscription_tool_call(
    *,
    tool_call: dict[str, Any],
    tool_runtime: ToolRuntime,
    tracker: RunTracker,
    source_url: str,
) -> tuple[str, str, dict[str, Any]]:
    tool_name = str(tool_call.get("name", "")).strip()
    raw_args = tool_call.get("args")
    tool_args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}

    if tool_name == "http_fetch":
        url = str(tool_args.get("url") or source_url)
        tracker.emit("ACTION", f"http_fetch args={{\"url\": \"{truncate_text(url, 220)}\"}}")
        observation_text = await tool_runtime.http_fetch(url)
        tracker.emit("OBSERVATION", f"http_fetch ok: {truncate_text(observation_text, 360)}")
        return observation_text, tool_name, tool_args

    if tool_name == "schema_inspect":
        tracker.emit("ACTION", "schema_inspect args={}")
        observation_text = tool_runtime.schema_inspect()
        tracker.emit("OBSERVATION", f"schema_inspect ok: {truncate_text(observation_text, 360)}")
        return observation_text, tool_name, tool_args

    if tool_name == "sql_exec":
        sql = str(tool_args.get("sql", "")).strip()
        if not sql:
            observation_text = json_dumps({"ok": False, "error": "Missing args.sql for sql_exec"})
            tracker.emit("OBSERVATION", "sql_exec error=Missing args.sql")
            return observation_text, tool_name, tool_args

        tracker.emit("SQL", f"[{classify_sql(sql)}] {truncate_text(sql, 1200)}")
        observation_text = tool_runtime.sql_exec(sql)
        emit_sql_observation(observation_text, tracker)
        return observation_text, tool_name, tool_args

    observation_text = json_dumps(
        {
            "ok": False,
            "error": (
                f"Unknown tool '{tool_name}'. "
                "Use one of: http_fetch, schema_inspect, sql_exec"
            ),
        }
    )
    tracker.emit("OBSERVATION", truncate_text(observation_text, 320))
    return observation_text, tool_name, tool_args


async def run_subscription_agent_loop(
    *,
    settings: Settings,
    tracker: RunTracker,
    tool_runtime: ToolRuntime,
    goal: str,
    source_url: str,
) -> None:
    history: list[dict[str, Any]] = []
    batch_tools_active = settings.batch_tools

    tracker.emit("THINK", "Subscription backend session started.")
    if settings.batch_tools:
        tracker.emit(
            "MODE",
            f"batch_tools=enabled max_batch_tool_calls={MAX_BATCH_TOOL_CALLS}",
        )

    for step_no in range(1, settings.max_tool_iterations + 1):
        tracker.step_count = step_no
        prompt = build_subscription_step_prompt(
            goal=goal,
            source_url=source_url,
            history=history,
            database_name=settings.database_name,
            batch_tools_enabled=batch_tools_active,
        )

        decision_started_at = time.perf_counter()
        decision_error: str | None = None
        try:
            raw_output = await request_subscription_decision(
                settings,
                prompt,
                batch_tools_enabled=batch_tools_active,
            )
        except Exception as exc:  # noqa: BLE001
            if batch_tools_active:
                batch_tools_active = False
                tracker.emit(
                    "MODE",
                    "batch_tools disabled: model decision failed; retrying this step in single-step mode.",
                )
                history.append(
                    {
                        "step": step_no,
                        "mode": "fallback",
                        "reason": summarize_observation_for_history(str(exc), 800),
                    }
                )
                try:
                    raw_output = await request_subscription_decision(
                        settings,
                        prompt,
                        batch_tools_enabled=False,
                    )
                except Exception as retry_exc:  # noqa: BLE001
                    decision_error = truncate_middle_text(str(retry_exc), 720)
                    decision_ms = int((time.perf_counter() - decision_started_at) * 1000)
                    tracker.model_decision_ms += decision_ms
                    tracker.append_step_perf(
                        {
                            "ts": utc_now_iso(),
                            "step_no": step_no,
                            "status": "decision_error",
                            "decision_ms": decision_ms,
                            "tool_name": None,
                            "tool_exec_ms": 0,
                            "db_exec_ms": 0,
                            "total_ms": decision_ms,
                            "error": decision_error,
                        }
                    )
                    raise
            else:
                decision_error = truncate_middle_text(str(exc), 720)
                decision_ms = int((time.perf_counter() - decision_started_at) * 1000)
                tracker.model_decision_ms += decision_ms
                tracker.append_step_perf(
                    {
                        "ts": utc_now_iso(),
                        "step_no": step_no,
                        "status": "decision_error",
                        "decision_ms": decision_ms,
                        "tool_name": None,
                        "tool_exec_ms": 0,
                        "db_exec_ms": 0,
                        "total_ms": decision_ms,
                        "error": decision_error,
                    }
                )
                raise

        decision_ms = int((time.perf_counter() - decision_started_at) * 1000)
        tracker.model_decision_ms += decision_ms
        parsed = extract_first_json_object(raw_output)
        tracker.append_raw_event(
            step_no,
            "subscription_decision",
            {
                "provider": settings.model_provider,
                "raw_output": summarize_observation_for_history(raw_output, 3000),
                "parsed": parsed,
            },
        )

        if not isinstance(parsed, dict):
            observation = (
                "Model output is not valid JSON; expected a single JSON object with tool_call/tool_calls/final_answer."
            )
            tracker.emit("OBSERVATION", observation)
            history.append({
                "step": step_no,
                "error": observation,
                "raw_output": summarize_observation_for_history(raw_output),
            })
            tracker.append_step_perf(
                {
                    "ts": utc_now_iso(),
                    "step_no": step_no,
                    "status": "invalid_json",
                    "decision_ms": decision_ms,
                    "tool_name": None,
                    "tool_exec_ms": 0,
                    "db_exec_ms": 0,
                    "total_ms": decision_ms,
                }
            )
            continue

        thought = str(parsed.get("thought", "")).strip()
        if thought:
            tracker.emit("THINK", truncate_text(thought, 1200))

        final_answer = parsed.get("final_answer")
        if isinstance(final_answer, str) and final_answer.strip():
            tracker.final_answer = final_answer.strip()
            tracker.emit("FINAL", truncate_text(tracker.final_answer, 2000))
            tracker.append_step_perf(
                {
                    "ts": utc_now_iso(),
                    "step_no": step_no,
                    "status": "final",
                    "decision_ms": decision_ms,
                    "tool_name": None,
                    "tool_exec_ms": 0,
                    "db_exec_ms": 0,
                    "total_ms": decision_ms,
                }
            )
            return

        selected_tool_calls, used_batch_response, parse_error = parse_subscription_tool_calls(
            parsed,
            batch_tools_enabled=batch_tools_active,
        )
        if parse_error:
            observation = parse_error
            tracker.emit("OBSERVATION", observation)
            history.append({"step": step_no, "error": observation})
            tracker.append_step_perf(
                {
                    "ts": utc_now_iso(),
                    "step_no": step_no,
                    "status": "missing_tool_call",
                    "decision_ms": decision_ms,
                    "tool_name": None,
                    "tool_exec_ms": 0,
                    "db_exec_ms": 0,
                    "total_ms": decision_ms,
                }
            )

            if used_batch_response and batch_tools_active:
                batch_tools_active = False
                tracker.emit(
                    "MODE",
                    "batch_tools disabled: invalid batch response format; falling back to single-step mode.",
                )
            continue

        batch_error: str | None = None
        batch_size = len(selected_tool_calls)

        for call_index, tool_call in enumerate(selected_tool_calls, start=1):
            tool_name = str(tool_call.get("name", "")).strip() or "unknown_tool"
            raw_args = tool_call.get("args")
            tool_args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
            observation_text = ""
            tool_error_text: str | None = None

            before_tool_exec_ms = tool_runtime.total_tool_exec_ms
            before_db_exec_ms = tool_runtime.total_db_exec_ms

            try:
                observation_text, tool_name, tool_args = await execute_subscription_tool_call(
                    tool_call=tool_call,
                    tool_runtime=tool_runtime,
                    tracker=tracker,
                    source_url=source_url,
                )
            except Exception as exc:  # noqa: BLE001
                tool_error_text = str(exc)
                observation_text = json_dumps({"ok": False, "error": tool_error_text})
                tracker.emit(
                    "OBSERVATION",
                    f"{tool_name} exception={truncate_text(tool_error_text, 360)}",
                )

            step_tool_exec_ms = max(tool_runtime.total_tool_exec_ms - before_tool_exec_ms, 0)
            step_db_exec_ms = max(tool_runtime.total_db_exec_ms - before_db_exec_ms, 0)
            decision_cost_ms = decision_ms if call_index == 1 else 0

            step_payload: dict[str, Any] = {
                "ts": utc_now_iso(),
                "step_no": step_no,
                "call_index": call_index,
                "batch_size": batch_size,
                "status": "tool_exception" if tool_error_text else "tool_called",
                "decision_ms": decision_cost_ms,
                "tool_name": tool_name,
                "tool_exec_ms": step_tool_exec_ms,
                "db_exec_ms": step_db_exec_ms,
                "total_ms": decision_cost_ms + step_tool_exec_ms,
            }

            if observation_text:
                parsed_observation = safe_json_loads(observation_text)
                if isinstance(parsed_observation, dict) and "ok" in parsed_observation:
                    step_payload["tool_ok"] = bool(parsed_observation.get("ok"))
            if tool_error_text:
                step_payload["error"] = truncate_text(tool_error_text, 360)
            tracker.append_step_perf(step_payload)

            history.append(
                {
                    "step": step_no,
                    "call_index": call_index,
                    "thought": thought,
                    "tool_call": {"name": tool_name, "args": tool_args},
                    "observation": summarize_observation_for_history(observation_text),
                }
            )

            call_failed = bool(tool_error_text) or subscription_observation_is_error(observation_text)
            if call_failed:
                batch_error = truncate_text(observation_text, 360)
                if not used_batch_response:
                    if tool_error_text:
                        raise RuntimeError(tool_error_text)
                    break
                break

        if used_batch_response and batch_error and batch_tools_active:
            batch_tools_active = False
            tracker.emit(
                "MODE",
                "batch_tools disabled after batch execution failure; falling back to single-step mode.",
            )
            history.append(
                {
                    "step": step_no,
                    "mode": "fallback",
                    "reason": summarize_observation_for_history(batch_error),
                }
            )
            continue

    if not tracker.final_answer:
        tracker.final_answer = (
            "Agent reached max tool iterations before producing a final answer. "
            "Increase MAX_TOOL_ITERATIONS and retry."
        )
        tracker.emit("FINAL", tracker.final_answer)


def normalize_openai_model_name(model_name: str) -> str:
    trimmed = model_name.strip()
    if trimmed.lower().startswith("openai/"):
        return trimmed.split("/", 1)[1]
    return trimmed


def infer_openai_model_info(model_name: str) -> dict[str, Any]:
    from autogen_ext.models.openai._model_info import get_info

    normalized = normalize_openai_model_name(model_name)
    try:
        info = get_info(normalized)
        if isinstance(info, dict):
            return dict(info)
    except Exception:  # noqa: BLE001
        pass

    lower = normalized.lower()
    if "gpt-5" in lower or "codex" in lower:
        family = "gpt-5"
    elif "gpt-4o" in lower:
        family = "gpt-4o"
    elif lower.startswith("o1"):
        family = "o1"
    elif lower.startswith("o3"):
        family = "o3"
    elif lower.startswith("o4"):
        family = "o4"
    elif "gpt-4" in lower:
        family = "gpt-4"
    else:
        family = "unknown"

    return {
        "vision": True,
        "function_calling": True,
        "json_output": True,
        "family": family,
        "structured_output": True,
        "multiple_system_messages": True,
    }


def supports_reasoning_effort(model_info: dict[str, Any]) -> bool:
    family = str(model_info.get("family", "")).lower()
    return family in {"gpt-5", "o1", "o3", "o4"}


def create_model_client(settings: Settings) -> Any:
    provider = settings.model_provider
    timeout_sec = float(settings.model_timeout_sec)

    if provider in {"openai", "openai_compatible", "gemini"}:
        if not settings.model_api_key:
            raise ValueError(
                f"MODEL_API_KEY is required for provider '{provider}'. "
                "Use codex_subscription/claude_subscription for subscription login modes."
            )
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        resolved_model_name = normalize_openai_model_name(settings.model_name)
        model_info = infer_openai_model_info(resolved_model_name)

        kwargs: dict[str, Any] = {
            "model": resolved_model_name,
            "api_key": settings.model_api_key,
            "timeout": timeout_sec,
            "max_retries": settings.model_max_retries,
            "model_info": model_info,
        }
        if settings.model_reasoning_effort and supports_reasoning_effort(model_info):
            kwargs["reasoning_effort"] = settings.model_reasoning_effort
        if settings.model_base_url:
            kwargs["base_url"] = settings.model_base_url
        if settings.model_organization:
            kwargs["organization"] = settings.model_organization
        return OpenAIChatCompletionClient(**kwargs)

    if provider == "anthropic":
        if not settings.model_api_key:
            raise ValueError(
                "MODEL_API_KEY (or ANTHROPIC_API_KEY) is required for provider 'anthropic'. "
                "Use claude_subscription if you want CLI subscription login mode."
            )
        from autogen_ext.models.anthropic import AnthropicChatCompletionClient

        kwargs = {
            "model": settings.model_name,
            "api_key": settings.model_api_key,
            "timeout": timeout_sec,
            "max_retries": settings.model_max_retries,
        }
        if settings.model_base_url:
            kwargs["base_url"] = settings.model_base_url
        return AnthropicChatCompletionClient(**kwargs)

    raise ValueError(f"Unsupported model provider: {provider}")


def make_run_artifacts(root: Path, run_id: str) -> RunArtifacts:
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        timeline_path=run_dir / "timeline.jsonl",
        events_path=run_dir / "events.jsonl",
        sql_audit_path=run_dir / "sql_audit.jsonl",
        step_perf_path=run_dir / "step_perf.jsonl",
        perf_path=run_dir / "perf.json",
        instance_path=run_dir / "tidb_zero_instance.json",
        final_path=run_dir / "final_answer.txt",
        error_trace_path=run_dir / "error_traceback.txt",
    )


def save_instance_credentials(artifacts: RunArtifacts, instance: ZeroInstance) -> None:
    payload = {
        "connectionString": instance.connection_string,
        "host": instance.host,
        "port": instance.port,
        "username": instance.username,
        "password": instance.password,
        "expiresAt": instance.expires_at,
        "savedAt": utc_now_iso(),
    }
    artifacts.instance_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.chmod(artifacts.instance_path, 0o600)


def extract_sql_argument(arguments: str) -> str:
    sql_value = extract_json_argument(arguments, "sql")
    if sql_value is not None:
        return normalize_sql(sql_value)
    return normalize_sql(arguments)


def extract_thought_argument(arguments: str) -> str:
    value = extract_json_argument(arguments, "thought")
    if value is not None:
        return value
    return arguments.strip()


def render_stream_event(event: Any, tracker: RunTracker) -> str | None:
    event_type = type(event).__name__

    if event_type == "ThoughtEvent":
        thought = str(getattr(event, "content", "")).strip()
        if thought:
            tracker.emit("THINK", truncate_text(thought, 1200))
        return None

    if event_type == "ToolCallRequestEvent":
        calls = getattr(event, "content", []) or []
        for call in calls:
            name = str(getattr(call, "name", "unknown_tool"))
            arguments = str(getattr(call, "arguments", ""))
            if name == "log_thought":
                thought = extract_thought_argument(arguments)
                tracker.emit("THINK", truncate_text(thought, 1200))
            elif name == "sql_exec":
                sql = extract_sql_argument(arguments)
                tracker.emit("SQL", f"[{classify_sql(sql)}] {truncate_text(sql, 1200)}")
            else:
                tracker.emit("ACTION", f"{name} args={truncate_text(arguments, 500)}")
        return None

    if event_type == "ToolCallExecutionEvent":
        results = getattr(event, "content", []) or []
        for result in results:
            name = str(getattr(result, "name", "unknown_tool"))
            content = str(getattr(result, "content", ""))
            is_error = bool(getattr(result, "is_error", False))

            if name == "log_thought":
                if is_error:
                    tracker.emit("OBSERVATION", f"log_thought error: {truncate_text(content, 360)}")
                continue

            if name == "sql_exec":
                parsed = safe_json_loads(content)
                if isinstance(parsed, dict):
                    statement_type = str(parsed.get("statement_type", "OTHER"))
                    ok = bool(parsed.get("ok")) and not is_error
                    raw_result = parsed.get("result")
                    result_payload: dict[str, Any] = raw_result if isinstance(raw_result, dict) else {}
                    if ok and result_payload.get("type") == "query":
                        rows = result_payload.get("row_count")
                        tracker.emit("OBSERVATION", f"sql_exec {statement_type} ok rows={rows}")
                    elif ok:
                        affected = result_payload.get("affected_rows")
                        tracker.emit("OBSERVATION", f"sql_exec {statement_type} ok affected_rows={affected}")
                    else:
                        error_text = str(parsed.get("error") or content)
                        tracker.emit("OBSERVATION", f"sql_exec {statement_type} error={truncate_text(error_text, 360)}")
                else:
                    tracker.emit("OBSERVATION", f"sql_exec raw={truncate_text(content, 360)}")
            else:
                status = "error" if is_error else "ok"
                tracker.emit("OBSERVATION", f"{name} {status}: {truncate_text(content, 360)}")
        return None

    if event_type == "TextMessage":
        source = str(getattr(event, "source", ""))
        if source == "user":
            return None

        text = str(getattr(event, "content", "")).strip()
        if not text:
            return None

        tracker.last_assistant_text = text
        if "TERMINATE" in text:
            final = text.replace("TERMINATE", "").strip()
            if final:
                tracker.final_answer = final
                tracker.emit("FINAL", truncate_text(final, 2000))
                return final
            return ""

        tracker.emit("THINK", truncate_text(text, 1200))
        return None

    return None


@dataclass(frozen=True)
class RunResult:
    run_id: str
    run_dir: Path
    final_answer: str


async def run_autonomous_demo(settings: Settings, goal: str, source_url: str) -> RunResult:
    run_started_at = time.perf_counter()
    zero_provision_ms = 0
    run_id = str(uuid.uuid4())
    artifacts = make_run_artifacts(settings.runs_dir, run_id)
    tracker = RunTracker(artifacts=artifacts)

    if is_subscription_provider(settings.model_provider):
        ensure_subscription_cli_available(settings)

    tracker.emit("TIDB_ZERO", "Provisioning TiDB Zero ephemeral instance...")
    zero_started_at = time.perf_counter()
    instance = provision_zero_instance(
        tag=settings.tidb_zero_tag,
        timeout_sec=settings.http_timeout_sec,
    )
    zero_provision_ms = int((time.perf_counter() - zero_started_at) * 1000)
    save_instance_credentials(artifacts, instance)

    tracker.emit(
        "TIDB_ZERO",
        (
            f"Ready host={instance.host}:{instance.port} expires_at={instance.expires_at} "
            f"quota_left={instance.remaining_database_quota}"
        ),
    )
    tracker.emit("TIDB_ZERO", f"Credentials saved to {artifacts.instance_path} (permissions 600)")

    tracker.emit(
        "MODEL",
        (
            f"provider={settings.model_provider} model={settings.model_name} "
            f"base_url={settings.model_base_url or 'default'} "
            f"timeout_sec={settings.model_timeout_sec} max_retries={settings.model_max_retries} "
            f"reasoning_effort={settings.model_reasoning_effort or 'default'}"
        ),
    )
    tracker.emit("MODE", f"batch_tools={settings.batch_tools}")

    sandbox = TiDBSandbox(instance=instance, database_name=settings.database_name)
    model_client: Any | None = None
    tool_runtime: ToolRuntime | None = None
    step_log_failed = False

    try:
        sandbox.create_database_if_missing()
        sandbox.connect()
        sandbox.initialize_metadata_tables()
        try:
            sandbox.log_run_start(
                run_id,
                goal,
                source_url,
                f"{settings.model_provider}:{settings.model_name}",
            )
        except Exception as log_exc:  # noqa: BLE001
            tracker.emit(
                "OBSERVATION",
                (
                    "run log start persistence failed; continuing run. "
                    f"reason={truncate_text(str(log_exc), 240)}"
                ),
            )

        tool_runtime = ToolRuntime(
            sandbox=sandbox,
            tracker=tracker,
            run_id=run_id,
            sql_row_limit=settings.sql_row_limit,
            fetch_max_chars=settings.fetch_max_chars,
            http_timeout_sec=settings.http_timeout_sec,
        )

        if is_subscription_provider(settings.model_provider):
            await run_subscription_agent_loop(
                settings=settings,
                tracker=tracker,
                tool_runtime=tool_runtime,
                goal=goal,
                source_url=source_url,
            )
        else:
            active_model_client = create_model_client(settings)
            model_client = active_model_client

            agent = AssistantAgent(
                name="autonomous_data_agent",
                model_client=active_model_client,
                tools=[
                    tool_runtime.log_thought,
                    tool_runtime.http_fetch,
                    tool_runtime.schema_inspect,
                    tool_runtime.sql_exec,
                ],
                system_message=build_system_message(settings.database_name),
                reflect_on_tool_use=True,
                max_tool_iterations=settings.max_tool_iterations,
                model_client_stream=True,
            )

            task = build_task(goal, source_url)
            tracker.emit("THINK", "Agent session started.")

            step_no = 0
            async for event in agent.run_stream(task=task):
                step_no += 1
                tracker.step_count = step_no
                event_type = type(event).__name__
                payload = summarize_event(event)
                tracker.append_raw_event(step_no, event_type, payload)

                try:
                    payload_text = json_dumps(payload)
                    if len(payload_text) > 64000:
                        payload_text = payload_text[:63980] + "...[truncated]"
                    sandbox.log_step(run_id, step_no, event_type, payload_text)
                except Exception as log_exc:  # noqa: BLE001
                    if not step_log_failed:
                        step_log_failed = True
                        tracker.emit(
                            "OBSERVATION",
                            (
                                "step log persistence failed; continuing run with local event file only. "
                                f"reason={truncate_text(str(log_exc), 240)}"
                            ),
                        )

                render_stream_event(event, tracker)

            if not tracker.final_answer:
                fallback = tracker.last_assistant_text.replace("TERMINATE", "").strip()
                tracker.final_answer = fallback
                if fallback:
                    tracker.emit("FINAL", truncate_text(fallback, 2000))

        proof_tables = sorted(tracker.created_tables)
        tracker.emit(
            "AUTONOMY_PROOF",
            (
                f"sql_count={tracker.sql_count} ddl_count={tracker.ddl_count} "
                f"tables_created={len(proof_tables)} {proof_tables}"
            ),
        )

        final_answer = tracker.final_answer or "Agent finished without a final text answer."
        artifacts.final_path.write_text(final_answer, encoding="utf-8")

        run_total_ms = int((time.perf_counter() - run_started_at) * 1000)
        total_tool_exec_ms = tool_runtime.total_tool_exec_ms if tool_runtime is not None else 0
        total_db_exec_ms = tool_runtime.total_db_exec_ms if tool_runtime is not None else 0
        tool_exec_ms_by_name = dict(tool_runtime.tool_exec_ms_by_name) if tool_runtime is not None else {}
        db_exec_ms_by_statement_type = dict(tool_runtime.db_exec_ms_by_statement_type) if tool_runtime is not None else {}
        db_exec_ms_by_task = dict(tool_runtime.db_exec_ms_by_task) if tool_runtime is not None else {}
        model_decision_ms = tracker.model_decision_ms
        if model_decision_ms <= 0 and not is_subscription_provider(settings.model_provider):
            derived_model_ms = run_total_ms - zero_provision_ms - total_tool_exec_ms
            model_decision_ms = max(derived_model_ms, 0)
        other_overhead_ms = max(run_total_ms - zero_provision_ms - model_decision_ms - total_tool_exec_ms, 0)
        perf_payload = {
            "run_id": run_id,
            "status": "completed",
            "provider": settings.model_provider,
            "model": settings.model_name,
            "batch_tools_enabled": settings.batch_tools,
            "run_total_ms": run_total_ms,
            "zero_provision_ms": zero_provision_ms,
            "model_decision_ms": model_decision_ms,
            "tool_exec_ms": total_tool_exec_ms,
            "db_exec_ms": total_db_exec_ms,
            "other_overhead_ms": other_overhead_ms,
            "tool_exec_ms_by_name": tool_exec_ms_by_name,
            "db_exec_ms_by_statement_type": db_exec_ms_by_statement_type,
            "db_exec_ms_by_task": db_exec_ms_by_task,
            "step_count": tracker.step_count,
            "sql_count": tracker.sql_count,
            "ddl_count": tracker.ddl_count,
        }
        artifacts.perf_path.write_text(json.dumps(perf_payload, indent=2), encoding="utf-8")
        tracker.emit(
            "TIME_TOTAL",
            f"overall={format_ms(run_total_ms)} ({run_total_ms}ms)",
        )
        tracker.emit(
            "TIME_PHASE",
            (
                f"tidb_zero={format_ms(zero_provision_ms)} ({zero_provision_ms}ms) "
                f"model_decision={format_ms(model_decision_ms)} ({model_decision_ms}ms) "
                f"tool_exec={format_ms(total_tool_exec_ms)} ({total_tool_exec_ms}ms) "
                f"db_exec={format_ms(total_db_exec_ms)} ({total_db_exec_ms}ms) "
                f"overhead={format_ms(other_overhead_ms)} ({other_overhead_ms}ms)"
            ),
        )
        tool_tasks_text = format_duration_map(tool_exec_ms_by_name)
        if tool_tasks_text:
            tracker.emit("TIME_TASKS", tool_tasks_text)
        sql_tasks_text = format_duration_map(db_exec_ms_by_statement_type)
        if sql_tasks_text:
            tracker.emit("TIME_SQL", sql_tasks_text)
        try:
            sandbox.log_run_end(run_id, final_answer, status="completed")
        except Exception as log_exc:  # noqa: BLE001
            tracker.emit(
                "OBSERVATION",
                (
                    "run completion log persistence failed; run result is still valid. "
                    f"reason={truncate_text(str(log_exc), 240)}"
                ),
            )

        return RunResult(run_id=run_id, run_dir=artifacts.run_dir, final_answer=final_answer)

    except Exception as exc:  # noqa: BLE001
        error_summary = f"Run failed: {exc}"
        tracker.emit("ERROR", error_summary)
        if is_timeout_like_error(exc):
            tracker.emit(
                "ERROR",
                (
                    "Model/API request timed out. Try increasing MODEL_TIMEOUT_SEC "
                    "(for example 120 or 180) and retry."
                ),
            )
        trace_text = traceback.format_exc()
        artifacts.error_trace_path.write_text(trace_text, encoding="utf-8")
        tracker.emit("ERROR", f"Traceback saved to {artifacts.error_trace_path}")

        run_total_ms = int((time.perf_counter() - run_started_at) * 1000)
        total_tool_exec_ms = tool_runtime.total_tool_exec_ms if tool_runtime is not None else 0
        total_db_exec_ms = tool_runtime.total_db_exec_ms if tool_runtime is not None else 0
        tool_exec_ms_by_name = dict(tool_runtime.tool_exec_ms_by_name) if tool_runtime is not None else {}
        db_exec_ms_by_statement_type = dict(tool_runtime.db_exec_ms_by_statement_type) if tool_runtime is not None else {}
        db_exec_ms_by_task = dict(tool_runtime.db_exec_ms_by_task) if tool_runtime is not None else {}
        model_decision_ms = tracker.model_decision_ms
        if model_decision_ms <= 0 and not is_subscription_provider(settings.model_provider):
            derived_model_ms = run_total_ms - zero_provision_ms - total_tool_exec_ms
            model_decision_ms = max(derived_model_ms, 0)
        other_overhead_ms = max(run_total_ms - zero_provision_ms - model_decision_ms - total_tool_exec_ms, 0)
        perf_payload = {
            "run_id": run_id,
            "status": "failed",
            "provider": settings.model_provider,
            "model": settings.model_name,
            "batch_tools_enabled": settings.batch_tools,
            "run_total_ms": run_total_ms,
            "zero_provision_ms": zero_provision_ms,
            "model_decision_ms": model_decision_ms,
            "tool_exec_ms": total_tool_exec_ms,
            "db_exec_ms": total_db_exec_ms,
            "other_overhead_ms": other_overhead_ms,
            "tool_exec_ms_by_name": tool_exec_ms_by_name,
            "db_exec_ms_by_statement_type": db_exec_ms_by_statement_type,
            "db_exec_ms_by_task": db_exec_ms_by_task,
            "step_count": tracker.step_count,
            "sql_count": tracker.sql_count,
            "ddl_count": tracker.ddl_count,
            "error": str(exc),
        }
        artifacts.perf_path.write_text(json.dumps(perf_payload, indent=2), encoding="utf-8")
        tracker.emit(
            "TIME_TOTAL",
            f"overall={format_ms(run_total_ms)} ({run_total_ms}ms)",
        )
        tracker.emit(
            "TIME_PHASE",
            (
                f"tidb_zero={format_ms(zero_provision_ms)} ({zero_provision_ms}ms) "
                f"model_decision={format_ms(model_decision_ms)} ({model_decision_ms}ms) "
                f"tool_exec={format_ms(total_tool_exec_ms)} ({total_tool_exec_ms}ms) "
                f"db_exec={format_ms(total_db_exec_ms)} ({total_db_exec_ms}ms) "
                f"overhead={format_ms(other_overhead_ms)} ({other_overhead_ms}ms)"
            ),
        )
        tool_tasks_text = format_duration_map(tool_exec_ms_by_name)
        if tool_tasks_text:
            tracker.emit("TIME_TASKS", tool_tasks_text)
        sql_tasks_text = format_duration_map(db_exec_ms_by_statement_type)
        if sql_tasks_text:
            tracker.emit("TIME_SQL", sql_tasks_text)

        if sandbox.is_connected:
            try:
                sandbox.log_run_end(run_id, error_summary, status="failed")
            except Exception:  # noqa: BLE001
                pass
        raise

    finally:
        if model_client is not None:
            await model_client.close()
        sandbox.close()


def replay_run(runs_dir: Path, run_id: str) -> None:
    timeline_path = runs_dir / run_id / "timeline.jsonl"
    if not timeline_path.exists():
        raise FileNotFoundError(f"Run timeline not found: {timeline_path}")

    print(f"[REPLAY] run_id={run_id}")
    with timeline_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = safe_json_loads(line)
            if not isinstance(item, dict):
                continue
            kind = item.get("kind", "EVENT")
            message = item.get("message", "")
            print(f"[{kind}] {message}")


def audit_run(runs_dir: Path, run_id: str) -> None:
    sql_audit_path = runs_dir / run_id / "sql_audit.jsonl"
    step_perf_path = runs_dir / run_id / "step_perf.jsonl"
    perf_path = runs_dir / run_id / "perf.json"
    if not sql_audit_path.exists():
        raise FileNotFoundError(f"SQL audit file not found: {sql_audit_path}")

    sql_count = 0
    ddl_count = 0
    error_count = 0
    created_tables: set[str] = set()

    print(f"[AUDIT] run_id={run_id}")
    with sql_audit_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = safe_json_loads(line)
            if not isinstance(item, dict):
                continue

            sql_count += 1
            statement_type = str(item.get("statement_type", "OTHER"))
            is_error = bool(item.get("is_error", False))
            if statement_type == "DDL":
                ddl_count += 1
            if is_error:
                error_count += 1

            sql_text = str(item.get("sql_text", ""))
            maybe_table = extract_created_table(sql_text)
            if maybe_table:
                created_tables.add(maybe_table)

            status = "ERROR" if is_error else "OK"
            elapsed_ms = item.get("elapsed_ms")
            print(
                f"[SQL_AUDIT] {statement_type} {status} elapsed_ms={elapsed_ms} sql={truncate_text(sql_text, 260)}"
            )

    print(
        "[AUTONOMY_PROOF] "
        f"sql_count={sql_count} ddl_count={ddl_count} error_count={error_count} "
        f"tables_created={len(created_tables)} {sorted(created_tables)}"
    )

    run_total_ms: int | None = None
    if perf_path.exists():
        perf_raw = safe_json_loads(perf_path.read_text(encoding="utf-8"))
        if isinstance(perf_raw, dict):
            raw_total = perf_raw.get("run_total_ms")
            run_total_ms = int(raw_total) if isinstance(raw_total, int) else None
            print(
                "[PERF] "
                f"run_total_ms={perf_raw.get('run_total_ms')} "
                f"zero_provision_ms={perf_raw.get('zero_provision_ms')} "
                f"model_decision_ms={perf_raw.get('model_decision_ms')} "
                f"tool_exec_ms={perf_raw.get('tool_exec_ms')} "
                f"db_exec_ms={perf_raw.get('db_exec_ms')} "
                f"step_count={perf_raw.get('step_count')}"
            )

            other_overhead = perf_raw.get("other_overhead_ms")
            if isinstance(other_overhead, int):
                print(
                    "[TIME_TOTAL] "
                    f"overall={format_ms(int(perf_raw.get('run_total_ms', 0) or 0))} "
                    f"overhead={format_ms(other_overhead)}"
                )

            raw_tool_map = perf_raw.get("tool_exec_ms_by_name")
            if isinstance(raw_tool_map, dict):
                tool_map = {str(k): int(v) for k, v in raw_tool_map.items() if isinstance(v, int)}
                tool_text = format_duration_map(tool_map)
                if tool_text:
                    print(f"[TIME_TASKS] {tool_text}")

            raw_sql_map = perf_raw.get("db_exec_ms_by_statement_type")
            if isinstance(raw_sql_map, dict):
                sql_map = {str(k): int(v) for k, v in raw_sql_map.items() if isinstance(v, int)}
                sql_text = format_duration_map(sql_map)
                if sql_text:
                    print(f"[TIME_SQL] {sql_text}")

    if step_perf_path.exists():
        step_records: list[dict[str, Any]] = []
        with step_perf_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = safe_json_loads(line)
                if isinstance(item, dict):
                    step_records.append(item)

        if step_records:
            print(f"[PERF_STEPS] count={len(step_records)} sorted_by=total_ms_desc")
            sorted_steps = sorted(
                step_records,
                key=lambda item: int(item.get("total_ms", 0) or 0),
                reverse=True,
            )
            for item in sorted_steps:
                total_ms = int(item.get("total_ms", 0) or 0)
                pct = ""
                if run_total_ms and run_total_ms > 0:
                    pct = f" pct_of_run={((total_ms / run_total_ms) * 100):.1f}%"
                call_suffix = ""
                call_index = item.get("call_index")
                batch_size = item.get("batch_size")
                if isinstance(call_index, int) and isinstance(batch_size, int):
                    call_suffix = f" call={call_index}/{batch_size}"
                print(
                    "[PERF_STEP] "
                    f"step={item.get('step_no')}{call_suffix} "
                    f"status={item.get('status')} "
                    f"tool={item.get('tool_name')} "
                    f"decision_ms={item.get('decision_ms')} "
                    f"tool_exec_ms={item.get('tool_exec_ms')} "
                    f"db_exec_ms={item.get('db_exec_ms')} "
                    f"total_ms={total_ms}{pct}"
                )


def show_tidb_zero_connection(runs_dir: Path, run_id: str, *, redact_password: bool = False) -> None:
    instance_path = runs_dir / run_id / "tidb_zero_instance.json"
    if not instance_path.exists():
        raise FileNotFoundError(f"TiDB Zero instance file not found: {instance_path}")

    raw = safe_json_loads(instance_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid TiDB Zero instance payload in: {instance_path}")

    connection_string = str(raw.get("connectionString", ""))
    host = str(raw.get("host", ""))
    port = str(raw.get("port", ""))
    username = str(raw.get("username", ""))
    password = str(raw.get("password", ""))
    expires_at = str(raw.get("expiresAt", ""))

    if not connection_string or not host or not port or not username or not password:
        raise ValueError(f"Incomplete TiDB Zero connection fields in: {instance_path}")

    shown_password = "***" if redact_password else password
    shown_connection_string = connection_string
    if redact_password:
        parsed = urlparse(connection_string)
        netloc = parsed.netloc
        if "@" in netloc:
            creds, host_part = netloc.split("@", 1)
            if ":" in creds:
                user, _ = creds.split(":", 1)
                netloc = f"{user}:***@{host_part}"
                shown_connection_string = parsed._replace(netloc=netloc).geturl()

    print(f"[TIDB_ZERO_CONNECTION] run_id={run_id}")
    print(f"connection_string={shown_connection_string}")
    print(f"host={host}")
    print(f"port={port}")
    print(f"username={username}")
    print(f"password={shown_password}")
    print(f"expires_at={expires_at}")
