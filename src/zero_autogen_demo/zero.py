from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

import httpx


ZERO_API_URL = "https://zero.tidbapi.com/v1alpha1/instances"


class ZeroProvisionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ZeroInstance:
    host: str
    port: int
    username: str
    password: str
    connection_string: str
    expires_at: str
    remaining_database_quota: int | None

    @property
    def redacted_connection_string(self) -> str:
        parsed = urlparse(self.connection_string)
        netloc = parsed.netloc
        if "@" in netloc:
            creds, host_port = netloc.split("@", 1)
            if ":" in creds:
                user, _ = creds.split(":", 1)
                creds = f"{user}:***"
            netloc = f"{creds}@{host_port}"
        return parsed._replace(netloc=netloc).geturl()


def provision_zero_instance(tag: str, timeout_sec: int = 20) -> ZeroInstance:
    payload = {
        "tag": tag,
    }

    try:
        response = httpx.post(ZERO_API_URL, json=payload, timeout=timeout_sec)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ZeroProvisionError(f"Failed to provision TiDB Zero instance: {exc}") from exc

    data = response.json()
    instance = data.get("instance") or {}
    connection = instance.get("connection") or {}

    host = connection.get("host")
    port = int(connection.get("port", 4000))
    username = connection.get("username")
    password = connection.get("password")
    connection_string = instance.get("connectionString")
    expires_at = instance.get("expiresAt")

    if not (host and username and password and connection_string and expires_at):
        raise ZeroProvisionError(
            "TiDB Zero API response is missing expected fields: "
            "instance.connection.{host,username,password}, instance.connectionString, instance.expiresAt"
        )

    return ZeroInstance(
        host=host,
        port=port,
        username=username,
        password=password,
        connection_string=connection_string,
        expires_at=expires_at,
        remaining_database_quota=data.get("remainingDatabaseQuota"),
    )
