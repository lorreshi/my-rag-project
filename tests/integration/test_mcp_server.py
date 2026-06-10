"""Integration tests for the MCP server (E1+).

Drives the server as a subprocess over stdio to verify the transport contract:
stdout carries only JSON-RPC messages, logs go to stderr, and initialize works.
In-process tests cover the server loop with mocked streams.
"""
from __future__ import annotations

import json
import subprocess
import sys
from io import StringIO

import pytest

from src.mcp_server.protocol_handler import ProtocolHandler
from src.mcp_server.server import MCPServer


def _make_request(method, params=None, req_id=1):
    msg = {"jsonrpc": "2.0", "method": method, "id": req_id}
    if params is not None:
        msg["params"] = params
    return json.dumps(msg)


class TestStdioSubprocess:
    """Black-box: run server as a subprocess, feed stdin, read stdout."""

    def test_initialize_over_stdio(self):
        request = _make_request("initialize", {"protocolVersion": "2025-06-18"})
        proc = subprocess.run(
            [sys.executable, "-m", "src.mcp_server.server"],
            input=request + "\n",
            capture_output=True,
            text=True,
            timeout=30,
        )
        # stdout must contain exactly one valid JSON-RPC response
        out_lines = [l for l in proc.stdout.splitlines() if l.strip()]
        assert len(out_lines) == 1
        response = json.loads(out_lines[0])
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert response["result"]["serverInfo"]["name"] == "smart-knowledge-hub"

    def test_stdout_not_polluted_by_logs(self):
        request = _make_request("initialize", {})
        proc = subprocess.run(
            [sys.executable, "-m", "src.mcp_server.server"],
            input=request + "\n",
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Every stdout line must be valid JSON (no log lines leaked)
        for line in proc.stdout.splitlines():
            if line.strip():
                json.loads(line)  # raises if a log line slipped in
        # Logs should appear on stderr
        assert "MCP server" in proc.stderr


class TestServerLoopInProcess:
    """White-box: drive the loop with in-memory streams."""

    def test_initialize_response(self):
        stdin = StringIO(_make_request("initialize", {}) + "\n")
        stdout = StringIO()
        MCPServer(stdin=stdin, stdout=stdout).serve_forever()
        out = [l for l in stdout.getvalue().splitlines() if l.strip()]
        resp = json.loads(out[0])
        assert resp["result"]["protocolVersion"]

    def test_blank_lines_ignored(self):
        stdin = StringIO("\n\n" + _make_request("initialize", {}) + "\n\n")
        stdout = StringIO()
        MCPServer(stdin=stdin, stdout=stdout).serve_forever()
        out = [l for l in stdout.getvalue().splitlines() if l.strip()]
        assert len(out) == 1

    def test_malformed_json_returns_parse_error(self):
        stdin = StringIO("{ this is not json\n")
        stdout = StringIO()
        MCPServer(stdin=stdin, stdout=stdout).serve_forever()
        resp = json.loads(stdout.getvalue().splitlines()[0])
        assert resp["error"]["code"] == -32700

    def test_notification_no_response(self):
        # notifications/initialized has no id -> no response written
        stdin = StringIO(
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        )
        stdout = StringIO()
        MCPServer(stdin=stdin, stdout=stdout).serve_forever()
        assert stdout.getvalue().strip() == ""

    def test_multiple_requests(self):
        stdin = StringIO(
            _make_request("initialize", {}, req_id=1) + "\n"
            + _make_request("tools/list", {}, req_id=2) + "\n"
        )
        stdout = StringIO()
        handler = ProtocolHandler()
        MCPServer(handler=handler, stdin=stdin, stdout=stdout).serve_forever()
        out = [json.loads(l) for l in stdout.getvalue().splitlines() if l.strip()]
        assert len(out) == 2
        assert out[0]["id"] == 1
        assert out[1]["id"] == 2
