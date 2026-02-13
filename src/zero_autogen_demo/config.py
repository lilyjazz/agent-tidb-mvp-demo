from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    tidb_zero_invitation_code: str
    model_name: str
    tidb_zero_tag: str
    database_name: str
    runs_dir: Path
    max_tool_iterations: int
    sql_row_limit: int
    http_timeout_sec: int
    fetch_max_chars: int

    @classmethod
    def from_env(
        cls,
        *,
        model_name: str | None = None,
        tidb_zero_tag: str | None = None,
        max_tool_iterations: int | None = None,
        sql_row_limit: int | None = None,
        http_timeout_sec: int | None = None,
        fetch_max_chars: int | None = None,
    ) -> "Settings":
        openai_api_key = _required_env("OPENAI_API_KEY")
        tidb_zero_invitation_code = _required_env("TIDB_ZERO_INVITATION_CODE")

        resolved_model = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
        resolved_tag = tidb_zero_tag or os.getenv("TIDB_ZERO_TAG", "agent-demo")
        resolved_db_name = os.getenv("TIDB_DB_NAME", "agent_sandbox")
        resolved_runs_dir = Path(os.getenv("RUNS_DIR", ".runs"))
        resolved_max_tool_iterations = max_tool_iterations or int(os.getenv("MAX_TOOL_ITERATIONS", "24"))
        resolved_sql_row_limit = sql_row_limit or int(os.getenv("SQL_ROW_LIMIT", "200"))
        resolved_http_timeout_sec = http_timeout_sec or int(os.getenv("HTTP_TIMEOUT_SEC", "20"))
        resolved_fetch_max_chars = fetch_max_chars or int(os.getenv("FETCH_MAX_CHARS", "60000"))

        return cls(
            openai_api_key=openai_api_key,
            tidb_zero_invitation_code=tidb_zero_invitation_code,
            model_name=resolved_model,
            tidb_zero_tag=resolved_tag,
            database_name=resolved_db_name,
            runs_dir=resolved_runs_dir,
            max_tool_iterations=resolved_max_tool_iterations,
            sql_row_limit=resolved_sql_row_limit,
            http_timeout_sec=resolved_http_timeout_sec,
            fetch_max_chars=resolved_fetch_max_chars,
        )
