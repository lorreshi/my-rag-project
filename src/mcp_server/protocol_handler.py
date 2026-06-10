"""JSON-RPC 2.0 protocol handler for the MCP server.

Handles the three core MCP methods:
- ``initialize``  -> server info + capabilities (capability negotiation)
- ``tools/list``  -> registered tool schemas
- ``tools/call``  -> route to a registered tool, return result or JSON-RPC error

Error codes follow JSON-RPC 2.0:
    -32700 Parse error
    -32600 Invalid Request
    -32601 Method not found
    -32602 Invalid params
    -32603 Internal error
"""
from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger("mcp-server")

# JSON-RPC 2.0 error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class ToolSpec:
    """A registered tool: schema + handler callable."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: Callable[[dict[str, Any]], dict[str, Any]],
    ):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.handler = handler

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


class ProtocolHandler:
    """Parse and dispatch JSON-RPC 2.0 requests for MCP."""

    def __init__(
        self,
        server_name: str = "smart-knowledge-hub",
        server_version: str = "0.1.0",
        protocol_version: str = "2025-06-18",
    ):
        self._server_name = server_name
        self._server_version = server_version
        self._protocol_version = protocol_version
        self._tools: dict[str, ToolSpec] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        """Register a tool with its schema and handler."""
        self._tools[name] = ToolSpec(name, description, input_schema, handler)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def handle(self, request: Any) -> dict[str, Any] | None:
        """Dispatch a parsed JSON-RPC request.

        Returns a response dict, or None for notifications (requests with no id).
        """
        if not isinstance(request, dict):
            return self._error(None, INVALID_REQUEST, "Request must be an object")

        jsonrpc = request.get("jsonrpc")
        method = request.get("method")
        req_id = request.get("id")
        params = request.get("params", {})

        is_notification = "id" not in request

        if jsonrpc != "2.0":
            if is_notification:
                return None
            return self._error(req_id, INVALID_REQUEST, "Invalid jsonrpc version")

        if not isinstance(method, str) or not method:
            if is_notification:
                return None
            return self._error(req_id, INVALID_REQUEST, "Missing method")

        try:
            if method == "initialize":
                result = self.handle_initialize(params)
            elif method == "notifications/initialized":
                # Client acknowledgement notification — no response.
                return None
            elif method == "tools/list":
                result = self.handle_tools_list()
            elif method == "tools/call":
                result = self.handle_tools_call(params)
            else:
                if is_notification:
                    return None
                return self._error(
                    req_id, METHOD_NOT_FOUND, f"Method not found: {method}"
                )
        except InvalidParams as exc:
            return self._error(req_id, INVALID_PARAMS, str(exc))
        except Exception as exc:  # internal error — do not leak stack trace
            logger.exception("Internal error handling %s", method)
            return self._error(req_id, INTERNAL_ERROR, "Internal error")

        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    # ------------------------------------------------------------------
    # Method implementations
    # ------------------------------------------------------------------

    def handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return server info + capabilities (capability negotiation)."""
        return {
            "protocolVersion": self._protocol_version,
            "serverInfo": {
                "name": self._server_name,
                "version": self._server_version,
            },
            "capabilities": {
                "tools": {"listChanged": False},
            },
        }

    def handle_tools_list(self) -> dict[str, Any]:
        """Return the schemas of all registered tools."""
        return {"tools": [spec.schema() for spec in self._tools.values()]}

    def handle_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        """Route a tools/call request to the registered tool handler."""
        if not isinstance(params, dict):
            raise InvalidParams("params must be an object")
        name = params.get("name")
        arguments = params.get("arguments", {})
        if not isinstance(name, str) or not name:
            raise InvalidParams("Missing tool name")
        if not isinstance(arguments, dict):
            raise InvalidParams("arguments must be an object")

        spec = self._tools.get(name)
        if spec is None:
            raise InvalidParams(f"Unknown tool: {name}")

        return spec.handler(arguments)

    # ------------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------------

    def parse_error(self, detail: str = "") -> dict[str, Any]:
        """Return a JSON-RPC parse error (id is null per spec)."""
        return self._error(None, PARSE_ERROR, "Parse error")

    @staticmethod
    def _error(req_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        }


class InvalidParams(Exception):
    """Raised by handlers when parameters are invalid (-> -32602)."""
