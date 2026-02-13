from __future__ import annotations

import json
import os
import re
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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def truncate_text(text: str, limit: int = 320) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 15] + "...[truncated]"


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

    def emit(self, kind: str, message: str) -> None:
        line = f"[{kind}] {message}"
        print(line, flush=True)
        self._append_jsonl(
            self.artifacts.timeline_path,
            {
                "ts": utc_now_iso(),
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

    async def http_fetch(self, url: str) -> str:
        """Fetch a public HTTP/HTTPS URL with GET and return status, headers, and body text."""
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

    def log_thought(self, thought: str) -> str:
        """Record a concise thought summary before taking an action."""
        cleaned = thought.strip()
        if not cleaned:
            return json_dumps({"ok": False, "error": "thought cannot be empty"})
        return json_dumps({"ok": True, "thought": truncate_text(cleaned, 800)})

    def schema_inspect(self) -> str:
        """Inspect current database schema and return tables/columns metadata as JSON."""
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
        result, _elapsed = self.sandbox.execute_sql(sql, max_rows=2000)
        return json_dumps(result)

    def sql_exec(self, sql: str) -> str:
        """Execute exactly one SQL statement against TiDB Zero and return JSON result."""
        normalized_sql = normalize_sql(sql)
        statement_type = classify_sql(normalized_sql)
        elapsed_ms = 0
        result_rows: int | None = None
        error_text: str | None = None

        try:
            result, elapsed_ms = self.sandbox.execute_sql(normalized_sql, max_rows=self.sql_row_limit)
            if result.get("type") == "query":
                result_rows = int(result.get("row_count", 0))
            else:
                affected = result.get("affected_rows")
                result_rows = int(affected) if isinstance(affected, int) else None
            ok = True
        except Exception as exc:  # noqa: BLE001
            result = {
                "type": "error",
                "error": str(exc),
            }
            error_text = str(exc)
            ok = False

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


def build_system_message(database_name: str) -> str:
    return (
        "You are an autonomous data agent operating inside a dedicated TiDB Zero database. "
        f"Your active schema is `{database_name}`. "
        "You have full SQL privileges in this schema. "
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
    run_id = str(uuid.uuid4())
    artifacts = make_run_artifacts(settings.runs_dir, run_id)
    tracker = RunTracker(artifacts=artifacts)

    tracker.emit("TIDB_ZERO", "Provisioning TiDB Zero ephemeral instance...")
    instance = provision_zero_instance(
        invitation_code=settings.tidb_zero_invitation_code,
        tag=settings.tidb_zero_tag,
        timeout_sec=settings.http_timeout_sec,
    )
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

    sandbox = TiDBSandbox(instance=instance, database_name=settings.database_name)
    model_client: Any | None = None
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

        model_client = create_model_client(settings)

        agent = AssistantAgent(
            name="autonomous_data_agent",
            model_client=model_client,
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
