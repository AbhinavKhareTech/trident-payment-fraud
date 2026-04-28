"""Live Razorpay MCP server client.

Implements the MCPServer ABC from protocol.py so it can be used
anywhere a food_server / instamart_server / dineout_server is used.
The mock-to-live swap is a config change, not a code change.

Remote MCP endpoint: https://mcp.razorpay.com/mcp
Auth: Basic Auth (key_id:key_secret, base64-encoded)
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any

import httpx

from bgi_trident.mcp.protocol import MCPServer, MCPToolResult

logger = logging.getLogger(__name__)

RAZORPAY_MCP_URL = os.getenv("RAZORPAY_MCP_URL", "https://mcp.razorpay.com/mcp")


def _auth_header(key_id: str, key_secret: str) -> str:
    token = base64.b64encode(f"{key_id}:{key_secret}".encode()).decode()
    return f"Basic {token}"


class RazorpayMCPServer(MCPServer):
    """Live Razorpay MCP server client.

    Wraps 35+ Razorpay payment tools through the MCP protocol.
    Used by RazorpayFraudAgent for live payment operations.

    The BGI risk tools (assess_payment_risk etc.) are on a SEPARATE
    server (MockBGIRiskServer or live BGI MCP).  RazorpayMCPServer
    only handles payment actions.
    """

    def __init__(
        self,
        key_id: str | None = None,
        key_secret: str | None = None,
        base_url: str = RAZORPAY_MCP_URL,
        timeout: float = 30.0,
    ) -> None:
        self._key_id = key_id or os.getenv("RAZORPAY_KEY_ID", "")
        self._key_secret = key_secret or os.getenv("RAZORPAY_KEY_SECRET", "")
        self._base_url = base_url
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        headers = {
            "Authorization": _auth_header(self._key_id, self._key_secret),
            "Content-Type": "application/json",
        }
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=self._timeout,
        )
        logger.info("RazorpayMCPServer connected to %s", self._base_url)

    async def call_tool(self, tool_name: str, params: dict[str, Any]) -> MCPToolResult:
        if self._client is None:
            raise RuntimeError("Not connected. Call connect() first.")
        try:
            response = await self._client.post(
                "/",
                json={"tool": tool_name, "arguments": params},
            )
            response.raise_for_status()
            data = response.json()
            return MCPToolResult(
                tool_name=tool_name,
                success=True,
                data=data,
            )
        except httpx.HTTPStatusError as e:
            logger.error("Razorpay MCP tool %s failed: %s", tool_name, e)
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                data={},
                error=str(e),
            )
        except Exception as e:
            logger.error("Unexpected error calling %s: %s", tool_name, e)
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                data={},
                error=str(e),
            )

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("RazorpayMCPServer connection closed")

    def available_tools(self) -> list[str]:
        """35+ Razorpay tools. Subset most relevant to fraud + risk workflows."""
        return [
            # Payments
            "fetch_payment",
            "capture_payment",
            "update_payment",
            "list_payments",
            # Orders
            "create_order",
            "fetch_order",
            "update_order",
            # Payment links
            "create_payment_link",
            "create_upi_payment_link",
            "send_payment_link",
            "fetch_payment_link",
            # Refunds
            "create_refund",
            "fetch_refund",
            "list_refunds",
            # Settlements
            "fetch_settlement",
            "list_settlements",
            # Disputes
            "fetch_dispute",
            "list_disputes",
        ]
