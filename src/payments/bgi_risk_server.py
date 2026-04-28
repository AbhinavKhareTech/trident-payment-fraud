"""BGI Risk MCP Server -- in-process mock for demo and testing.

In production this would be the live BGI Trident MCP server
(FastMCP server over stdio or SSE transport).

For demo, it wraps the PaymentGraphBuilder directly so there is
no network hop, but the interface is identical to the live server:
it implements MCPServer and exposes the same three tools.

Swap to live server:  replace BGIRiskMCPServer with
a thin httpx client pointing at the deployed BGI Trident endpoint.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from bgi_trident.mcp.protocol import MCPServer, MCPToolResult

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


class BGIRiskMCPServer(MCPServer):
    """In-process BGI Trident fraud intelligence server.

    Loads the payment graph from CSV fixtures and exposes:
      - assess_payment_risk
      - detect_merchant_ring
      - generate_dispute_evidence

    The graph is loaded once at connect() time.
    """

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self._data_dir = Path(data_dir) if data_dir else _DATA_DIR
        self._engine   = None     # PaymentRiskEngine loaded at connect()

    async def connect(self) -> None:
        from bgi_trident.mcp.bgi_risk_engine import PaymentRiskEngine
        self._engine = PaymentRiskEngine(self._data_dir)
        self._engine.load()
        logger.info("BGIRiskMCPServer connected (graph loaded)")

    async def call_tool(self, tool_name: str, params: dict[str, Any]) -> MCPToolResult:
        if self._engine is None:
            return MCPToolResult(
                tool_name=tool_name, success=False, data={},
                error="BGI engine not loaded. Call connect() first.",
            )
        try:
            handler = {
                "assess_payment_risk":      self._engine.assess_payment_risk,
                "detect_merchant_ring":     self._engine.detect_merchant_ring,
                "generate_dispute_evidence":self._engine.generate_dispute_evidence,
            }.get(tool_name)

            if handler is None:
                return MCPToolResult(
                    tool_name=tool_name, success=False, data={},
                    error=f"Unknown BGI tool: {tool_name}",
                )

            result = handler(**params)
            return MCPToolResult(tool_name=tool_name, success=True, data=result)

        except Exception as e:
            logger.error("BGI tool %s failed: %s", tool_name, e)
            return MCPToolResult(
                tool_name=tool_name, success=False, data={}, error=str(e),
            )

    async def close(self) -> None:
        self._engine = None

    def available_tools(self) -> list[str]:
        return ["assess_payment_risk", "detect_merchant_ring", "generate_dispute_evidence"]
