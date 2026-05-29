from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

from rlm_harness.config import CONFIG_DIR

MCP_CONFIG_PATH = Path(os.environ.get("HARNESS_MCP_CONFIG", CONFIG_DIR / "mcp.json"))
MCP_TRANSPORTS = {"stdio", "http", "sse"}
MCP_AUTH_TYPES = {"none", "bearer_env", "api_key_env", "oauth"}


@dataclass(frozen=True)
class MCPAuthConfig:
    type: str = "none"
    token_env: Optional[str] = None
    api_key_env: Optional[str] = None
    api_key_header: str = "x-api-key"
    scopes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.type not in MCP_AUTH_TYPES:
            raise ValueError(f"unsupported MCP auth type: {self.type}")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> MCPAuthConfig:
        if not isinstance(data, dict):
            return cls()
        return cls(
            type=str(data.get("type") or "none"),
            token_env=optional_str(data.get("token_env")),
            api_key_env=optional_str(data.get("api_key_env")),
            api_key_header=str(data.get("api_key_header") or "x-api-key"),
            scopes=list_of_strings(data.get("scopes")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "token_env": self.token_env,
            "api_key_env": self.api_key_env,
            "api_key_header": self.api_key_header,
            "scopes": list(self.scopes),
        }

    @property
    def is_authenticated(self) -> bool:
        return self.type != "none"

    @property
    def credential_env(self) -> Optional[str]:
        if self.type == "bearer_env":
            return self.token_env
        if self.type == "api_key_env":
            return self.api_key_env
        return None

    @property
    def credential_available(self) -> bool:
        env_name = self.credential_env
        if not env_name:
            return not self.is_authenticated
        return bool(os.environ.get(env_name))

    def summary(self) -> str:
        if self.type == "none":
            return "unauthenticated"
        env_name = self.credential_env
        if env_name:
            status = "present" if self.credential_available else "missing"
            return f"{self.type} via ${env_name} ({status})"
        if self.type == "oauth":
            scopes = ", ".join(self.scopes) if self.scopes else "default scopes"
            return f"oauth ({scopes})"
        return self.type

    def request_headers(self) -> dict[str, str]:
        if self.type == "none":
            return {}
        if self.type == "bearer_env":
            token = os.environ.get(self.token_env or "")
            if not token:
                raise ValueError(f"missing MCP bearer token env: {self.token_env}")
            return {"authorization": f"Bearer {token}"}
        if self.type == "api_key_env":
            api_key = os.environ.get(self.api_key_env or "")
            if not api_key:
                raise ValueError(f"missing MCP API key env: {self.api_key_env}")
            return {self.api_key_header: api_key}
        if self.type == "oauth":
            raise ValueError(
                "OAuth MCP re-auth is not implemented; use bearer_env for stored tokens"
            )
        return {}


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    transport: str
    url: Optional[str] = None
    command: Optional[str] = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    auth: MCPAuthConfig = field(default_factory=MCPAuthConfig)
    purposes: list[str] = field(default_factory=list)
    enabled: bool = True
    trusted: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("MCP server name is required")
        if self.transport not in MCP_TRANSPORTS:
            raise ValueError(f"unsupported MCP transport: {self.transport}")
        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio MCP servers require a command")
        if self.transport in {"http", "sse"} and not self.url:
            raise ValueError(f"{self.transport} MCP servers require a URL")

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> MCPServerConfig:
        return cls(
            name=name,
            transport=str(data.get("transport") or "stdio"),
            url=optional_str(data.get("url")),
            command=optional_str(data.get("command")),
            args=list_of_strings(data.get("args")),
            env=dict_of_strings(data.get("env")),
            headers=dict_of_strings(data.get("headers")),
            auth=MCPAuthConfig.from_dict(data.get("auth")),
            purposes=normalize_purposes(data.get("purposes")),
            enabled=bool(data.get("enabled", True)),
            trusted=bool(data.get("trusted", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "transport": self.transport,
            "url": self.url,
            "command": self.command,
            "args": list(self.args),
            "env": dict(self.env),
            "headers": dict(self.headers),
            "auth": self.auth.to_dict(),
            "purposes": list(self.purposes),
            "enabled": self.enabled,
            "trusted": self.trusted,
        }

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "transport": self.transport,
            "url": self.url,
            "command": self.command,
            "args": list(self.args),
            "purposes": list(self.purposes),
            "enabled": self.enabled,
            "trusted": self.trusted,
            "auth": self.auth.summary(),
        }

    def matches_task(self, task: str) -> bool:
        if not self.enabled:
            return False
        if not self.purposes:
            return False
        normalized_task = task.lower()
        return any(purpose.lower() in normalized_task for purpose in self.purposes)

    def matches_purpose(self, purpose: str | None) -> bool:
        if not purpose:
            return False
        normalized = purpose.lower()
        return self.enabled and any(item.lower() == normalized for item in self.purposes)

    def request_headers(self) -> dict[str, str]:
        headers = {key.lower(): value for key, value in self.headers.items()}
        headers.update(self.auth.request_headers())
        return headers


class MCPConfigStore:
    def __init__(self, path: Path | None = None):
        self.path = path or MCP_CONFIG_PATH

    def load(self) -> dict[str, MCPServerConfig]:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        servers = data.get("servers") if isinstance(data, dict) else None
        if not isinstance(servers, dict):
            return {}
        loaded = {}
        for name, payload in servers.items():
            if isinstance(payload, dict):
                try:
                    loaded[str(name)] = MCPServerConfig.from_dict(str(name), payload)
                except ValueError:
                    continue
        return loaded

    def save(self, servers: dict[str, MCPServerConfig]) -> None:
        payload = {
            "servers": {
                name: server.to_dict()
                for name, server in sorted(servers.items(), key=lambda item: item[0])
            }
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    def list(self) -> list[MCPServerConfig]:
        return list(self.load().values())

    def get(self, name: str) -> Optional[MCPServerConfig]:
        return self.load().get(name)

    def select(self, name: str | None = None, purpose: str | None = None) -> MCPServerConfig:
        servers = self.load()
        if name:
            server = servers.get(name)
            if server is None:
                raise ValueError(f"unknown MCP server: {name}")
            if not server.enabled:
                raise ValueError(f"MCP server is disabled: {name}")
            return server
        if not purpose:
            enabled = [server for server in servers.values() if server.enabled]
            if not enabled:
                raise ValueError("no enabled MCP servers configured")
            if len(enabled) > 1:
                names = ", ".join(server.name for server in enabled)
                raise ValueError(
                    f"multiple MCP servers configured; specify server or purpose: {names}"
                )
            return enabled[0]
        matches = [server for server in servers.values() if server.matches_purpose(purpose)]
        if not matches:
            raise ValueError(f"no enabled MCP server matches purpose: {purpose}")
        if len(matches) > 1:
            names = ", ".join(server.name for server in matches)
            raise ValueError(f"multiple MCP servers match purpose {purpose}: {names}")
        return matches[0]

    def add(self, server: MCPServerConfig) -> None:
        servers = self.load()
        servers[server.name] = server
        self.save(servers)

    def update(self, name: str, **changes: Any) -> Optional[MCPServerConfig]:
        servers = self.load()
        server = servers.get(name)
        if server is None:
            return None
        updated = replace(server, **changes)
        servers[name] = updated
        self.save(servers)
        return updated

    def remove(self, name: str) -> bool:
        servers = self.load()
        if name not in servers:
            return False
        del servers[name]
        self.save(servers)
        return True


def mcp_workflow_context(task: str, store: MCPConfigStore | None = None) -> str:
    servers = (store or MCPConfigStore()).list()
    enabled = [server for server in servers if server.enabled]
    if not enabled:
        return ""
    selected = [server for server in enabled if server.matches_task(task)]
    lines = []
    if selected:
        lines.append("MCP servers selected for this task:")
        lines.extend(render_mcp_context_lines(selected))
        selected_names = {item.name for item in selected}
        remaining = [server for server in enabled if server.name not in selected_names]
        if remaining:
            lines.extend(["", "Other configured MCP purpose routes:"])
            lines.extend(render_mcp_context_lines(remaining))
    else:
        lines.append("MCP servers available for designated workflow purposes:")
        lines.extend(render_mcp_context_lines(enabled))
    lines.append(
        "Use mcp_list_tools and mcp_call_tool with the matching server or purpose when "
        "the task needs that designated external system and credentials are present; "
        "do not improvise unrelated external access."
    )
    return "\n\n" + "\n".join(lines)


def render_mcp_context_lines(servers: list[MCPServerConfig]) -> list[str]:
    lines = []
    for server in servers:
        auth_status = server.auth.summary()
        trust = "trusted" if server.trusted else "approval-gated"
        purposes = ", ".join(server.purposes) or "unspecified"
        endpoint = server.url or " ".join([server.command or "", *server.args]).strip()
        lines.append(
            f"- {server.name}: {server.transport} {endpoint}; purposes={purposes}; "
            f"auth={auth_status}; {trust}"
        )
    return lines


def render_mcp_catalog(servers: list[MCPServerConfig]) -> str:
    if not servers:
        return "No MCP servers configured."
    lines = ["Harness MCP servers", ""]
    for server in servers:
        enabled = "enabled" if server.enabled else "disabled"
        trusted = "trusted" if server.trusted else "approval-gated"
        purposes = ", ".join(server.purposes) if server.purposes else "-"
        lines.append(f"{server.name}\t{server.transport}\t{enabled}\t{trusted}")
        lines.append(f"  purposes\t{purposes}")
        lines.append(f"  auth\t{server.auth.summary()}")
        if server.url:
            lines.append(f"  url\t{server.url}")
        if server.command:
            command = " ".join([server.command, *server.args]).strip()
            lines.append(f"  command\t{command}")
        lines.append("")
    return "\n".join(lines).rstrip()


def parse_key_values(values: list[str] | None, *, option: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values or []:
        key, separator, raw = value.partition("=")
        if not key or separator != "=":
            raise ValueError(f"{option} values must use KEY=VALUE syntax")
        parsed[key] = raw
    return parsed


def optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def dict_of_strings(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def normalize_purposes(value: Any) -> list[str]:
    purposes = list_of_strings(value)
    normalized = []
    for purpose in purposes:
        cleaned = purpose.strip().lower()
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized
