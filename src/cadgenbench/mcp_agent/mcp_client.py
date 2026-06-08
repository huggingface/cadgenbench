# Copyright 2026 Hugging Face
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Synchronous wrapper around a build123d-mcp stdio server.

Architecture:
  McpSession (main thread)
    └── background daemon thread running an asyncio event loop
         └── mcp.ClientSession ←─── stdio ───→ build123d-mcp subprocess

The MCP session (and its subprocess) stays alive for the full agent run.
Tool calls are dispatched via asyncio.run_coroutine_threadsafe() so the
main thread stays blocking-sync while the async MCP protocol runs in the
background loop.
"""
from __future__ import annotations

import asyncio
import base64
import threading
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class McpSession:
    """Long-lived synchronous proxy for a build123d-mcp MCP server.

    Usage::

        with McpSession("build123d-mcp") as session:
            text, png = session.call_tool("execute", {"code": "..."})
            tools = session.tools  # LiteLLM-format tool defs
    """

    def __init__(self, command: str, args: list[str] | None = None) -> None:
        self._params = StdioServerParameters(command=command, args=args or [])
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: ClientSession | None = None
        self._shutdown: asyncio.Event | None = None
        self._ready = threading.Event()
        self._init_error: Exception | None = None
        self._tools: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background event loop and connect to the MCP server."""
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="mcp-event-loop"
        )
        self._thread.start()
        if not self._ready.wait(timeout=90):
            raise TimeoutError("MCP server did not become ready within 90 s")
        if self._init_error is not None:
            raise self._init_error

    def close(self) -> None:
        """Signal shutdown and wait for the background thread to exit."""
        if self._loop is not None and self._shutdown is not None:
            self._loop.call_soon_threadsafe(self._shutdown.set)
        if self._thread is not None:
            self._thread.join(timeout=15)

    def __enter__(self) -> "McpSession":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Tool introspection
    # ------------------------------------------------------------------

    @property
    def tools(self) -> list[dict[str, Any]]:
        """LiteLLM (OpenAI-format) tool definitions from the MCP server."""
        return self._tools

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[str, bytes | None]:
        """Call an MCP tool synchronously.

        Returns ``(text_result, png_bytes)`` where ``png_bytes`` is the
        first PNG image in the tool result, or ``None`` if no image was
        returned.
        """
        if self._loop is None or self._session is None:
            raise RuntimeError("McpSession has not been started")
        future = asyncio.run_coroutine_threadsafe(
            self._call_async(name, arguments), self._loop
        )
        return future.result(timeout=300)

    # ------------------------------------------------------------------
    # Private async helpers (run in background loop)
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_session())
        finally:
            self._loop.close()

    async def _run_session(self) -> None:
        self._shutdown = asyncio.Event()
        try:
            async with stdio_client(self._params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    resp = await session.list_tools()
                    self._tools = _convert_tools(resp.tools)
                    self._session = session
                    self._ready.set()
                    await self._shutdown.wait()
        except Exception as exc:  # noqa: BLE001
            self._init_error = exc
            self._ready.set()

    async def _call_async(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[str, bytes | None]:
        result = await self._session.call_tool(name, arguments)  # type: ignore[union-attr]
        texts: list[str] = []
        png: bytes | None = None
        for block in result.content:
            if block.type == "text":
                texts.append(block.text)
            elif (
                block.type == "image"
                and getattr(block, "mimeType", "") == "image/png"
                and png is None
            ):
                png = base64.b64decode(block.data)
        return "\n".join(texts), png


# ---------------------------------------------------------------------------
# Schema conversion
# ---------------------------------------------------------------------------

def _convert_tools(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    """Convert MCP tool objects to LiteLLM (OpenAI-format) tool defs.

    The first sentence of the description is kept to stay within provider
    token limits for tool definitions. ``$schema`` is stripped because some
    providers reject it as an unrecognised keyword.
    """
    result = []
    for t in mcp_tools:
        schema = dict(t.inputSchema or {"type": "object", "properties": {}})
        schema.pop("$schema", None)
        desc = (t.description or "").split("\n")[0][:512]
        result.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": desc,
                "parameters": schema,
            },
        })
    return result
