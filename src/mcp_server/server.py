"""MCP Server entry point (Stdio Transport).

Implements the stdio transport contract for MCP:
- Reads newline-delimited JSON-RPC 2.0 messages from stdin.
- Writes newline-delimited JSON-RPC 2.0 responses to stdout.
- stdout carries ONLY protocol messages; all logging goes to stderr.

The actual request dispatch is delegated to a ProtocolHandler (E2). E1 wires
the transport loop and guarantees the stdout/stderr separation.
"""
from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any, TextIO

from src.observability.logger import get_logger

if TYPE_CHECKING:
    from src.mcp_server.protocol_handler import ProtocolHandler

logger = get_logger("mcp-server")

# MCP protocol version this server speaks.
PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "smart-knowledge-hub"
SERVER_VERSION = "0.1.0"


class MCPServer:
    """Stdio-transport MCP server."""

    def __init__(
        self,
        handler: "ProtocolHandler | None" = None,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
    ):
        """Initialize the server.

        Args:
            handler: ProtocolHandler for dispatch. A default is built lazily.
            stdin: Input stream (defaults to sys.stdin).
            stdout: Output stream (defaults to sys.stdout). MUST receive only
                protocol messages.
        """
        self._stdin = stdin or sys.stdin
        self._stdout = stdout or sys.stdout
        if handler is None:
            from src.mcp_server.protocol_handler import ProtocolHandler
            handler = ProtocolHandler(
                server_name=SERVER_NAME,
                server_version=SERVER_VERSION,
                protocol_version=PROTOCOL_VERSION,
            )
        self._handler = handler

    def _write_message(self, message: dict[str, Any]) -> None:
        """Write a single JSON-RPC message to stdout (newline-delimited)."""
        line = json.dumps(message, ensure_ascii=False)
        self._stdout.write(line + "\n")
        self._stdout.flush()

    def serve_forever(self) -> None:
        """Run the stdio read/dispatch/write loop until EOF."""
        logger.info(
            "MCP server starting (protocol=%s, server=%s/%s)",
            PROTOCOL_VERSION, SERVER_NAME, SERVER_VERSION,
        )
        for raw_line in self._stdin:
            line = raw_line.strip()
            if not line:
                continue
            response = self._process_line(line)
            # Notifications (no id) produce no response.
            if response is not None:
                self._write_message(response)
        logger.info("MCP server stdin closed; shutting down")

    def _process_line(self, line: str) -> dict[str, Any] | None:
        """Parse one line and dispatch it. Returns a response dict or None."""
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.error("Received malformed JSON: %s", exc)
            return self._handler.parse_error(str(exc))

        return self._handler.handle(request)


def _build_default_handler() -> "ProtocolHandler":
    """Build a ProtocolHandler with the built-in tools registered.

    Tool registration is best-effort: failures are logged to stderr and the
    server still starts (see ``register_default_tools``).
    """
    from src.mcp_server.protocol_handler import ProtocolHandler
    from src.mcp_server.tools.registry import register_default_tools

    handler = ProtocolHandler(
        server_name=SERVER_NAME,
        server_version=SERVER_VERSION,
        protocol_version=PROTOCOL_VERSION,
    )
    try:
        register_default_tools(handler)
    except Exception:  # config load failure etc. — start with no tools
        logger.exception("Tool registration failed; starting with no tools")
    return handler


def main() -> int:
    """Console entry point."""
    server = MCPServer(handler=_build_default_handler())
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover
        logger.info("Interrupted; exiting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
