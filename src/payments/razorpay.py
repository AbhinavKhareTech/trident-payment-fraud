"""Razorpay payments domain agent with BGI Trident fraud intelligence.

Extends BaseAgent (agents/base.py) exactly as FoodAgent, InstamartAgent,
and DineoutAgent do.  Wraps the RazorpayMCPServer (or MockRazorpayMCP)
and integrates BGI Trident risk assessment as a confirmation gate.

The key design decision: this agent calls BGI risk tools BEFORE every
transactional action (capture_payment, create_payment_link).  If BGI
returns BLOCK, the transaction never reaches Razorpay.

The orchestrator (PaymentRiskCoordinator) treats this the same way
TridentCoordinator treats food/instamart/dineout agents.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from bgi_trident.agents.base import BaseAgent
from bgi_trident.mcp.protocol import MCPServer, MCPToolResult

logger = logging.getLogger(__name__)


class RazorpayFraudAgent(BaseAgent):
    """Razorpay payments agent with integrated BGI Trident fraud gate.

    Two servers:
      - razorpay_server: MCPServer for Razorpay payment actions
      - bgi_server:      MCPServer for BGI Trident risk assessment

    The agent routes every transactional call through BGI first.
    """

    def __init__(
        self,
        razorpay_server: MCPServer,
        bgi_server:      MCPServer,
    ) -> None:
        super().__init__(razorpay_server)
        self.bgi_server = bgi_server

    # ── BaseAgent contract ────────────────────────────────────────────────────

    async def search(self, query: str, **kwargs: Any) -> MCPToolResult:
        """Search/list payments. Read-only; no risk gate needed."""
        return await self.server.call_tool("list_payments", {"query": query, **kwargs})

    async def add_to_cart(self, entity_id: str, **kwargs: Any) -> MCPToolResult:
        """Not applicable for payments; mirrors BaseAgent contract."""
        return MCPToolResult(
            tool_name="add_to_cart",
            success=False,
            data={},
            error="Payments domain does not use a cart. Use create_payment_link instead.",
        )

    async def execute_order(self, **kwargs: Any) -> MCPToolResult:
        """Capture a payment -- always gated through BGI risk assessment."""
        payment_id  = kwargs.get("payment_id",  "")
        merchant_id = kwargs.get("merchant_id", "")
        payer_id    = kwargs.get("payer_id",    "")
        amount      = kwargs.get("amount",      0.0)

        risk = await self._assess_risk(payment_id, merchant_id, payer_id, amount)
        if not risk.success:
            return risk

        decision = risk.data.get("decision", "REVIEW")
        if decision == "BLOCK":
            return MCPToolResult(
                tool_name="capture_payment",
                success=False,
                data={"bgi_decision": risk.data},
                error=f"BGI Trident BLOCKED this transaction. Score={risk.data.get('ensemble_score')}",
            )

        if decision == "REVIEW":
            logger.warning(
                "BGI Trident REVIEW for payment %s (score=%s). Proceeding with flag.",
                payment_id, risk.data.get("ensemble_score"),
            )

        return await self.server.call_tool("capture_payment", kwargs)

    # ── Domain-specific methods ───────────────────────────────────────────────

    async def create_payment_link(
        self,
        merchant_id: str,
        payer_id:    str,
        amount:      float,
        description: str = "",
        **kwargs: Any,
    ) -> MCPToolResult:
        """Create a Razorpay payment link -- gated through BGI risk."""
        # Synthetic payment ID for pre-auth assessment
        pre_auth_id = f"pre_{merchant_id[:6]}_{payer_id[:6]}"

        risk = await self._assess_risk(pre_auth_id, merchant_id, payer_id, amount)
        if risk.success and risk.data.get("decision") == "BLOCK":
            return MCPToolResult(
                tool_name="create_payment_link",
                success=False,
                data={"bgi_decision": risk.data},
                error=(
                    f"BGI Trident BLOCKED payment link creation.\n"
                    f"Score: {risk.data.get('ensemble_score')}\n"
                    f"Reason: {risk.data.get('graph_signals', [])}"
                ),
            )

        return await self.server.call_tool("create_payment_link", {
            "amount":      int(amount * 100),  # Razorpay expects paise
            "currency":    "INR",
            "description": description,
            **kwargs,
        })

    async def generate_dispute_evidence(
        self,
        payment_id:  str,
        dispute_id:  str,
        merchant_id: str,
        payer_id:    str,
        amount:      float,
        reason:      str = "customer_dispute",
    ) -> MCPToolResult:
        """Call BGI to auto-generate dispute evidence, then attach to Razorpay."""
        # Step 1: BGI generates evidence
        evidence = await self.bgi_server.call_tool("generate_dispute_evidence", {
            "payment_id":  payment_id,
            "dispute_id":  dispute_id,
            "merchant_id": merchant_id,
            "payer_id":    payer_id,
            "amount":      amount,
            "reason":      reason,
        })

        if not evidence.success:
            return evidence

        # Step 2: Fetch the dispute from Razorpay to confirm it exists
        dispute = await self.server.call_tool("fetch_dispute", {
            "dispute_id": dispute_id,
            "payment_id": payment_id,
            "amount":     amount,
        })

        return MCPToolResult(
            tool_name="generate_dispute_evidence",
            success=True,
            data={
                "bgi_evidence":    evidence.data,
                "razorpay_dispute": dispute.data,
                "ready_to_submit": evidence.data.get("evidence_strength") in ("STRONG", "MODERATE"),
            },
        )

    async def detect_merchant_ring(
        self,
        merchant_id:       str,
        min_shared_payers: int = 3,
    ) -> MCPToolResult:
        """Delegate ring analysis to BGI server directly."""
        return await self.bgi_server.call_tool("detect_merchant_ring", {
            "merchant_id":       merchant_id,
            "min_shared_payers": min_shared_payers,
        })

    # ── Private ───────────────────────────────────────────────────────────────

    async def _assess_risk(
        self,
        payment_id:  str,
        merchant_id: str,
        payer_id:    str,
        amount:      float,
    ) -> MCPToolResult:
        """Call BGI assess_payment_risk and return result."""
        logger.info(
            "BGI risk assessment: payment=%s merchant=%s payer=%s amount=%.2f",
            payment_id, merchant_id, payer_id, amount,
        )
        return await self.bgi_server.call_tool("assess_payment_risk", {
            "payment_id":  payment_id,
            "merchant_id": merchant_id,
            "payer_id":    payer_id,
            "amount":      amount,
        })

    async def connect(self) -> None:
        await self.server.connect()
        await self.bgi_server.connect()

    async def close(self) -> None:
        await self.server.close()
        await self.bgi_server.close()
