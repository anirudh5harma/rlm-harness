from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from rlm_harness import __version__
from rlm_harness.mcp_config import MCPServerConfig

MCP_PROTOCOL_VERSION = "2025-06-18"


class MCPClientError(RuntimeError):
    pass


@dataclass
class MCPClient:
    server: MCPServerConfig
    timeout_s: float = 30.0

    def list_tools(self) -> dict[str, Any]:
        with self._transport() as transport:
            transport.initialize()
            return transport.request("tools/list", {})

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._transport() as transport:
            transport.initialize()
            return transport.request(
                "tools/call",
                {"name": name, "arguments": arguments or {}},
            )

    def _transport(self) -> MCPTransport:
        if self.server.transport == "stdio":
            return StdioMCPTransport(self.server, self.timeout_s)
        if self.server.transport in {"http", "sse"}:
            return HttpMCPTransport(self.server, self.timeout_s)
        raise MCPClientError(f"unsupported MCP transport: {self.server.transport}")


class MCPTransport:
    def __enter__(self) -> MCPTransport:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def initialize(self) -> None:
        self.request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "rlm-harness", "version": __version__},
            },
        )
        self.notify("notifications/initialized", {})

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def notify(self, method: str, params: dict[str, Any]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class JsonRpcMixin:
    def __init__(self) -> None:
        self._next_id = 1

    def next_id(self) -> int:
        request_id = self._next_id
        self._next_id += 1
        return request_id

    @staticmethod
    def request_payload(request_id: int, method: str, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

    @staticmethod
    def notification_payload(method: str, params: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "method": method, "params": params}

    @staticmethod
    def result_from_response(response: dict[str, Any], request_id: int) -> dict[str, Any]:
        if response.get("id") != request_id:
            raise MCPClientError(f"unexpected MCP response id: {response.get('id')}")
        if response.get("error"):
            raise MCPClientError(json.dumps(response["error"], sort_keys=True))
        result = response.get("result")
        return result if isinstance(result, dict) else {"value": result}


class StdioMCPTransport(MCPTransport, JsonRpcMixin):
    def __init__(self, server: MCPServerConfig, timeout_s: float):
        JsonRpcMixin.__init__(self)
        self.server = server
        self.timeout_s = timeout_s
        env = os.environ.copy()
        env.update(server.env)
        if server.command is None:
            raise MCPClientError("stdio MCP server has no command")
        self.process = subprocess.Popen(
            [server.command, *server.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        self._stdout_lines: queue.Queue[str | None] = queue.Queue()
        self._stderr_tail: list[str] = []
        self._stderr_lock = threading.Lock()
        if self.process.stdout is not None:
            threading.Thread(target=self._read_stdout_loop, daemon=True).start()
        if self.process.stderr is not None:
            threading.Thread(target=self._read_stderr_loop, daemon=True).start()

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self.next_id()
        self._write(self.request_payload(request_id, method, params))
        while True:
            response = self._read()
            if response.get("id") == request_id:
                return self.result_from_response(response, request_id)

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._write(self.notification_payload(method, params))

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def _write(self, payload: dict[str, Any]) -> None:
        if self.process.stdin is None:
            raise MCPClientError("MCP stdio stdin is closed")
        try:
            self.process.stdin.write(json.dumps(payload) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise MCPClientError(f"MCP stdio write failed: {self._stderr_text()}") from exc

    def _read(self) -> dict[str, Any]:
        try:
            line = self._stdout_lines.get(timeout=self.timeout_s)
        except queue.Empty as exc:
            raise MCPClientError(
                f"MCP stdio timed out after {self.timeout_s:g}s waiting for response"
            ) from exc
        if line is None:
            raise MCPClientError(f"MCP stdio server exited: {self._stderr_text()}")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MCPClientError(f"invalid MCP stdio JSON: {line.strip()}") from exc
        if not isinstance(response, dict):
            raise MCPClientError("MCP stdio response was not an object")
        return response

    def _read_stdout_loop(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self._stdout_lines.put(line)
        self._stdout_lines.put(None)

    def _read_stderr_loop(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            with self._stderr_lock:
                self._stderr_tail.append(line.rstrip())
                self._stderr_tail = self._stderr_tail[-20:]

    def _stderr_text(self) -> str:
        with self._stderr_lock:
            text = "\n".join(self._stderr_tail).strip()
        return text or "no stderr"


class HttpMCPTransport(MCPTransport, JsonRpcMixin):
    def __init__(self, server: MCPServerConfig, timeout_s: float):
        JsonRpcMixin.__init__(self)
        self.server = server
        self.timeout_s = timeout_s
        self.protocol_version = MCP_PROTOCOL_VERSION
        self.session_id: str | None = None
        if server.url is None:
            raise MCPClientError("HTTP MCP server has no URL")

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self.next_id()
        response = self._post(self.request_payload(request_id, method, params))
        result = self.result_from_response(response, request_id)
        if method == "initialize":
            self.protocol_version = str(result.get("protocolVersion") or self.protocol_version)
        return result

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._post(self.notification_payload(method, params), expect_response=False)

    def _post(self, payload: dict[str, Any], *, expect_response: bool = True) -> dict[str, Any]:
        if self.server.url is None:
            raise MCPClientError("HTTP MCP server has no URL")
        headers = {
            "content-type": "application/json",
            "accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": self.protocol_version,
            **self.server.request_headers(),
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.server.url,
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = response.read().decode("utf-8")
                session_id = response.headers.get("Mcp-Session-Id")
                if session_id:
                    self.session_id = session_id
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise MCPClientError(f"MCP HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise MCPClientError(f"MCP HTTP request failed: {exc.reason}") from exc
        if not body.strip():
            return {} if not expect_response else {"jsonrpc": "2.0", "id": payload.get("id")}
        response_payload = parse_http_response(body)
        if not isinstance(response_payload, dict):
            raise MCPClientError("MCP HTTP response was not an object")
        return response_payload


def parse_http_response(body: str) -> Any:
    stripped = body.strip()
    if stripped.startswith("event:") or "\ndata:" in stripped:
        for line in stripped.splitlines():
            if line.startswith("data:"):
                return json.loads(line.removeprefix("data:").strip())
    return json.loads(stripped)
