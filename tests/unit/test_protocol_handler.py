"""Unit tests for ProtocolHandler — JSON-RPC 2.0 dispatch + error codes."""
from __future__ import annotations

import pytest

from src.mcp_server.protocol_handler import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    InvalidParams,
    ProtocolHandler,
)


def _req(method, params=None, req_id=1):
    msg = {"jsonrpc": "2.0", "method": method, "id": req_id}
    if params is not None:
        msg["params"] = params
    return msg


@pytest.fixture
def handler():
    h = ProtocolHandler(server_name="test-hub", server_version="9.9.9")
    return h


@pytest.fixture
def handler_with_tool(handler):
    handler.register_tool(
        name="echo",
        description="Echo the input back.",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        handler=lambda args: {"content": [{"type": "text", "text": args.get("text", "")}]},
    )
    return handler


class TestInitialize:
    def test_returns_server_info(self, handler):
        resp = handler.handle(_req("initialize", {}))
        assert resp["result"]["serverInfo"]["name"] == "test-hub"
        assert resp["result"]["serverInfo"]["version"] == "9.9.9"

    def test_returns_capabilities(self, handler):
        resp = handler.handle(_req("initialize", {}))
        assert "tools" in resp["result"]["capabilities"]

    def test_returns_protocol_version(self, handler):
        resp = handler.handle(_req("initialize", {}))
        assert resp["result"]["protocolVersion"]

    def test_response_envelope(self, handler):
        resp = handler.handle(_req("initialize", {}, req_id=42))
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 42


class TestToolsList:
    def test_empty_when_no_tools(self, handler):
        resp = handler.handle(_req("tools/list"))
        assert resp["result"]["tools"] == []

    def test_lists_registered_tool(self, handler_with_tool):
        resp = handler_with_tool.handle(_req("tools/list"))
        tools = resp["result"]["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == "echo"
        assert "inputSchema" in tools[0]
        assert tools[0]["description"]


class TestToolsCall:
    def test_routes_to_handler(self, handler_with_tool):
        resp = handler_with_tool.handle(
            _req("tools/call", {"name": "echo", "arguments": {"text": "hi"}})
        )
        assert resp["result"]["content"][0]["text"] == "hi"

    def test_unknown_tool_invalid_params(self, handler_with_tool):
        resp = handler_with_tool.handle(
            _req("tools/call", {"name": "nope", "arguments": {}})
        )
        assert resp["error"]["code"] == INVALID_PARAMS

    def test_missing_name_invalid_params(self, handler_with_tool):
        resp = handler_with_tool.handle(_req("tools/call", {"arguments": {}}))
        assert resp["error"]["code"] == INVALID_PARAMS

    def test_bad_arguments_type(self, handler_with_tool):
        resp = handler_with_tool.handle(
            _req("tools/call", {"name": "echo", "arguments": "not-a-dict"})
        )
        assert resp["error"]["code"] == INVALID_PARAMS

    def test_handler_invalid_params_propagates(self, handler):
        def _bad(args):
            raise InvalidParams("need a query")

        handler.register_tool("q", "d", {}, _bad)
        resp = handler.handle(_req("tools/call", {"name": "q", "arguments": {}}))
        assert resp["error"]["code"] == INVALID_PARAMS
        assert "need a query" in resp["error"]["message"]

    def test_internal_error_does_not_leak_stack(self, handler):
        def _boom(args):
            raise RuntimeError("secret internal detail")

        handler.register_tool("boom", "d", {}, _boom)
        resp = handler.handle(_req("tools/call", {"name": "boom", "arguments": {}}))
        assert resp["error"]["code"] == INTERNAL_ERROR
        assert "secret internal detail" not in resp["error"]["message"]
        assert resp["error"]["message"] == "Internal error"


class TestErrorHandling:
    def test_method_not_found(self, handler):
        resp = handler.handle(_req("does/not/exist"))
        assert resp["error"]["code"] == METHOD_NOT_FOUND

    def test_invalid_jsonrpc_version(self, handler):
        resp = handler.handle({"jsonrpc": "1.0", "method": "initialize", "id": 1})
        assert resp["error"]["code"] == INVALID_REQUEST

    def test_missing_method(self, handler):
        resp = handler.handle({"jsonrpc": "2.0", "id": 1})
        assert resp["error"]["code"] == INVALID_REQUEST

    def test_non_dict_request(self, handler):
        resp = handler.handle(["not", "a", "dict"])
        assert resp["error"]["code"] == INVALID_REQUEST

    def test_parse_error_helper(self, handler):
        resp = handler.parse_error("bad json")
        assert resp["error"]["code"] == PARSE_ERROR
        assert resp["id"] is None


class TestNotifications:
    def test_notification_no_response(self, handler):
        # no "id" key -> notification
        resp = handler.handle({"jsonrpc": "2.0", "method": "initialize"})
        assert resp is None

    def test_initialized_notification(self, handler):
        resp = handler.handle(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}
        )
        assert resp is None

    def test_unknown_notification_silent(self, handler):
        resp = handler.handle({"jsonrpc": "2.0", "method": "unknown/thing"})
        assert resp is None
