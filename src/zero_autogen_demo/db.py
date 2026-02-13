from __future__ import annotations

import json
import time
from datetime import date, datetime, time as datetime_time
from decimal import Decimal
from typing import Any

import certifi
import pymysql
from pymysql.cursors import DictCursor

from .zero import ZeroInstance


def json_dumps(value: Any) -> str:
    def _default(obj: Any) -> str:
        if isinstance(obj, (datetime, date, datetime_time, Decimal)):
            return str(obj)
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        return str(obj)

    return json.dumps(value, default=_default, ensure_ascii=True)


class TiDBSandbox:
    def __init__(self, instance: ZeroInstance, database_name: str) -> None:
        self._instance = instance
        self.database_name = database_name
        self._conn: pymysql.connections.Connection | None = None

    def _connect(self, database: str | None) -> pymysql.connections.Connection:
        params: dict[str, Any] = {
            "host": self._instance.host,
            "port": self._instance.port,
            "user": self._instance.username,
            "password": self._instance.password,
            "autocommit": True,
            "charset": "utf8mb4",
            "cursorclass": DictCursor,
            "connect_timeout": 10,
            "read_timeout": 60,
            "write_timeout": 60,
            "ssl": {"ca": certifi.where()},
        }
        if database:
            params["database"] = database
        return pymysql.connect(**params)

    def create_database_if_missing(self) -> None:
        admin_conn = self._connect(None)
        try:
            with admin_conn.cursor() as cur:
                cur.execute(f"CREATE DATABASE IF NOT EXISTS `{self.database_name}`")
        finally:
            admin_conn.close()

    def connect(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
        self._conn = self._connect(self.database_name)

    def _ensure_connection(self) -> None:
        if self._conn is None:
            self.connect()
            return

        try:
            self._conn.ping(reconnect=True)
        except Exception:  # noqa: BLE001
            self.connect()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def is_connected(self) -> bool:
        return self._conn is not None

    def execute_sql(self, sql: str, max_rows: int = 200) -> tuple[dict[str, Any], int]:
        self._ensure_connection()
        if self._conn is None:
            raise RuntimeError("Database is not connected")

        start = time.perf_counter()
        with self._conn.cursor() as cur:
            cur.execute(sql)
            elapsed_ms = int((time.perf_counter() - start) * 1000)

            if cur.description:
                rows = list(cur.fetchmany(max_rows + 1))
                truncated = len(rows) > max_rows
                if truncated:
                    rows = rows[:max_rows]
                columns = [column[0] for column in cur.description]
                return (
                    {
                        "type": "query",
                        "columns": columns,
                        "row_count": len(rows),
                        "truncated": truncated,
                        "rows": rows,
                    },
                    elapsed_ms,
                )

            return ({"type": "statement", "affected_rows": cur.rowcount}, elapsed_ms)

    def initialize_metadata_tables(self) -> None:
        ddl_statements = [
            """
            CREATE TABLE IF NOT EXISTS run_logs (
              run_id CHAR(36) PRIMARY KEY,
              goal TEXT NOT NULL,
              source_url TEXT NOT NULL,
              model_name VARCHAR(128) NOT NULL,
              final_answer LONGTEXT NULL,
              status VARCHAR(32) NOT NULL DEFAULT 'running',
              created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
              completed_at DATETIME(6) NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS step_logs (
              id BIGINT PRIMARY KEY AUTO_INCREMENT,
              run_id CHAR(36) NOT NULL,
              step_no INT NOT NULL,
              event_type VARCHAR(64) NOT NULL,
              payload LONGTEXT NULL,
              created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
              KEY idx_run_step (run_id, step_no)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS sql_audit (
              id BIGINT PRIMARY KEY AUTO_INCREMENT,
              run_id CHAR(36) NOT NULL,
              statement_type VARCHAR(16) NOT NULL,
              sql_text LONGTEXT NOT NULL,
              is_error BOOLEAN NOT NULL,
              result_rows BIGINT NULL,
              elapsed_ms INT NULL,
              error_text TEXT NULL,
              created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
              KEY idx_run_time (run_id, created_at)
            )
            """,
        ]
        for ddl in ddl_statements:
            self.execute_sql(ddl)

    def log_run_start(self, run_id: str, goal: str, source_url: str, model_name: str) -> None:
        self._ensure_connection()
        if self._conn is None:
            return
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO run_logs (run_id, goal, source_url, model_name, status)
                VALUES (%s, %s, %s, %s, 'running')
                """,
                (run_id, goal, source_url, model_name),
            )

    def log_run_end(self, run_id: str, final_answer: str, status: str) -> None:
        self._ensure_connection()
        if self._conn is None:
            return
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE run_logs
                SET final_answer = %s,
                    status = %s,
                    completed_at = CURRENT_TIMESTAMP(6)
                WHERE run_id = %s
                """,
                (final_answer, status, run_id),
            )

    def log_step(self, run_id: str, step_no: int, event_type: str, payload: str) -> None:
        self._ensure_connection()
        if self._conn is None:
            return
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO step_logs (run_id, step_no, event_type, payload)
                VALUES (%s, %s, %s, %s)
                """,
                (run_id, step_no, event_type, payload),
            )

    def log_sql_audit(
        self,
        *,
        run_id: str,
        statement_type: str,
        sql_text: str,
        is_error: bool,
        result_rows: int | None,
        elapsed_ms: int,
        error_text: str | None,
    ) -> None:
        self._ensure_connection()
        if self._conn is None:
            return
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sql_audit
                    (run_id, statement_type, sql_text, is_error, result_rows, elapsed_ms, error_text)
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s)
                """,
                (run_id, statement_type, sql_text, is_error, result_rows, elapsed_ms, error_text),
            )
