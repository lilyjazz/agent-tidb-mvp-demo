from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_MODEL_PROVIDERS = {"openai", "anthropic", "gemini", "openai_compatible"}


def _optional_env(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _normalize_provider(raw: str) -> str:
    provider = raw.strip().lower()
    aliases = {
        "claude": "anthropic",
        "openai-compatible": "openai_compatible",
        "openai_compat": "openai_compatible",
    }
    provider = aliases.get(provider, provider)
    if provider not in SUPPORTED_MODEL_PROVIDERS:
        expected = ", ".join(sorted(SUPPORTED_MODEL_PROVIDERS))
        raise ValueError(f"Unsupported MODEL_PROVIDER '{raw}'. Supported values: {expected}")
    return provider


def _default_model_for_provider(provider: str) -> str:
    defaults = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-sonnet-latest",
        "gemini": "gemini-2.0-flash",
        "openai_compatible": "gpt-4o-mini",
    }
    return defaults[provider]


def _resolve_model_api_key(provider: str, explicit_key: str | None) -> str:
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()

    generic_key = _optional_env("MODEL_API_KEY")
    if generic_key:
        return generic_key

    provider_candidates: dict[str, list[str]] = {
        "openai": ["OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
        "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "openai_compatible": ["OPENAI_API_KEY"],
    }

    for env_name in provider_candidates.get(provider, []):
        value = _optional_env(env_name)
        if value:
            return value

    fallback = _optional_env("OPENAI_API_KEY")
    if fallback:
        return fallback

    hints = {
        "openai": "MODEL_API_KEY or OPENAI_API_KEY",
        "anthropic": "MODEL_API_KEY or ANTHROPIC_API_KEY",
        "gemini": "MODEL_API_KEY or GEMINI_API_KEY",
        "openai_compatible": "MODEL_API_KEY",
    }
    hint = hints.get(provider, "MODEL_API_KEY")
    raise ValueError(f"Missing model API key for provider '{provider}'. Set {hint}.")


def _resolve_model_base_url(provider: str, explicit_base_url: str | None) -> str | None:
    if explicit_base_url and explicit_base_url.strip():
        return explicit_base_url.strip()

    env_base_url = _optional_env("MODEL_BASE_URL")
    if env_base_url:
        return env_base_url

    if provider == "gemini":
        return "https://generativelanguage.googleapis.com/v1beta/openai/"

    return None


@dataclass(frozen=True)
class Settings:
    model_provider: str
    model_api_key: str
    model_base_url: str | None
    model_organization: str | None
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
        model_provider: str | None = None,
        model_name: str | None = None,
        model_base_url: str | None = None,
        model_organization: str | None = None,
        model_api_key: str | None = None,
        tidb_zero_tag: str | None = None,
        max_tool_iterations: int | None = None,
        sql_row_limit: int | None = None,
        http_timeout_sec: int | None = None,
        fetch_max_chars: int | None = None,
    ) -> "Settings":
        resolved_provider = _normalize_provider(model_provider or os.getenv("MODEL_PROVIDER", "openai"))
        resolved_model = model_name or os.getenv("MODEL_NAME", _default_model_for_provider(resolved_provider))
        resolved_model_api_key = _resolve_model_api_key(resolved_provider, model_api_key)
        resolved_model_base_url = _resolve_model_base_url(resolved_provider, model_base_url)
        resolved_model_organization = model_organization or _optional_env("MODEL_ORGANIZATION")
        tidb_zero_invitation_code = _required_env("TIDB_ZERO_INVITATION_CODE")

        resolved_tag = tidb_zero_tag or os.getenv("TIDB_ZERO_TAG", "agent-demo")
        resolved_db_name = os.getenv("TIDB_DB_NAME", "agent_sandbox")
        resolved_runs_dir = Path(os.getenv("RUNS_DIR", ".runs"))
        resolved_max_tool_iterations = max_tool_iterations or int(os.getenv("MAX_TOOL_ITERATIONS", "24"))
        resolved_sql_row_limit = sql_row_limit or int(os.getenv("SQL_ROW_LIMIT", "200"))
        resolved_http_timeout_sec = http_timeout_sec or int(os.getenv("HTTP_TIMEOUT_SEC", "20"))
        resolved_fetch_max_chars = fetch_max_chars or int(os.getenv("FETCH_MAX_CHARS", "60000"))

        return cls(
            model_provider=resolved_provider,
            model_api_key=resolved_model_api_key,
            model_base_url=resolved_model_base_url,
            model_organization=resolved_model_organization,
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
