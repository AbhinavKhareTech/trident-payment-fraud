"""Mock Razorpay MCP server for demo and testing.

Mirrors MockFoodMCP / MockInstamartMCP from the mock/ directory.
Loads fixture data from fixtures/razorpay_merchants.json and
fixtures/razorpay_transactions.json.

The mock-to-live swap is: replace MockRazorpayMCP with RazorpayMCPServer.
"""

from __future__ import annotations

import json
import random
import string
from pathlib import Path
from typing import Any

from bgi_trident.mcp.protocol import MCPServer, MCPToolResult

FIXTURES = Path(__file__).parent / "fixtures"


def _random_id(prefix: str) -> str:
    suffix = "".join(random.choices(string.ascii_letters + string.digits, k=8))
    return f"{prefix}_{suffix}"


class MockRazorpayMCP(MCPServer):
    """Mock Razorpay MCP server.

    Supports the subset of tools used by RazorpayFraudAgent:
    create_payment_link, capture_payment, create_refund,
    fetch_payment, fetch_dispute.
    """

    def __init__(self) -> None:
        self._merchants:    list[dict] = []
        self._transactions: list[dict] = []
        self._payment_links: dict[str, dict] = {}
        self._payments:      dict[str, dict] = {}
        self._refunds:       dict[str, dict] = {}

    async def connect(self) -> None:
        merchants_path = FIXTURES / "razorpay_merchants.json"
        txns_path      = FIXTURES / "razorpay_transactions.json"

        if merchants_path.exists():
            self._merchants = json.loads(merchants_path.read_text())
        if txns_path.exists():
            self._transactions = json.loads(txns_path.read_text())
            # Index as payments dict for fetch
            for t in self._transactions:
                self._payments[t["payment_id"]] = t

    async def call_tool(self, tool_name: str, params: dict[str, Any]) -> MCPToolResult:
        handler = getattr(self, f"_handle_{tool_name}", None)
        if handler is None:
            return MCPToolResult(
                tool_name=tool_name, success=False, data={},
                error=f"Unknown tool: {tool_name}",
            )
        return await handler(params)

    async def _handle_create_payment_link(self, params: dict) -> MCPToolResult:
        link_id = _random_id("plink")
        link = {
            "id":          link_id,
            "amount":      params.get("amount", 0),
            "currency":    params.get("currency", "INR"),
            "description": params.get("description", ""),
            "short_url":   f"https://rzp.io/i/{link_id[:8]}",
            "status":      "created",
        }
        self._payment_links[link_id] = link
        return MCPToolResult(tool_name="create_payment_link", success=True, data=link)

    async def _handle_create_upi_payment_link(self, params: dict) -> MCPToolResult:
        return await self._handle_create_payment_link(params)

    async def _handle_capture_payment(self, params: dict) -> MCPToolResult:
        payment_id = params.get("payment_id", _random_id("pay"))
        payment = self._payments.get(payment_id, {
            "payment_id": payment_id,
            "amount":     params.get("amount", 0),
            "status":     "captured",
            "method":     params.get("method", "upi"),
        })
        payment["status"] = "captured"
        self._payments[payment_id] = payment
        return MCPToolResult(tool_name="capture_payment", success=True, data=payment)

    async def _handle_fetch_payment(self, params: dict) -> MCPToolResult:
        payment_id = params.get("payment_id", "")
        payment = self._payments.get(payment_id, {"error": "payment not found"})
        return MCPToolResult(
            tool_name="fetch_payment",
            success=bool(payment.get("payment_id")),
            data=payment,
        )

    async def _handle_create_refund(self, params: dict) -> MCPToolResult:
        refund_id = _random_id("rfnd")
        refund = {
            "id":         refund_id,
            "payment_id": params.get("payment_id"),
            "amount":     params.get("amount", 0),
            "status":     "processed",
            "speed":      params.get("speed", "normal"),
        }
        self._refunds[refund_id] = refund
        return MCPToolResult(tool_name="create_refund", success=True, data=refund)

    async def _handle_fetch_dispute(self, params: dict) -> MCPToolResult:
        dispute_id = params.get("dispute_id", "")
        # Return a synthetic dispute for demo
        return MCPToolResult(
            tool_name="fetch_dispute",
            success=True,
            data={
                "id":         dispute_id,
                "payment_id": params.get("payment_id", ""),
                "amount":     params.get("amount", 0),
                "reason":     "customer_dispute",
                "status":     "open",
            },
        )

    async def _handle_list_payments(self, params: dict) -> MCPToolResult:
        limit = params.get("count", 10)
        return MCPToolResult(
            tool_name="list_payments",
            success=True,
            data={"payments": self._transactions[:limit], "count": min(limit, len(self._transactions))},
        )

    async def close(self) -> None:
        pass

    def available_tools(self) -> list[str]:
        return [
            "create_payment_link", "create_upi_payment_link",
            "capture_payment", "fetch_payment", "list_payments",
            "create_refund", "fetch_dispute",
        ]
