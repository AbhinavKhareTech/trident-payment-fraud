"""Payment Risk Coordinator.

Extends the TridentCoordinator pattern (orchestrator/coordinator.py)
for the Razorpay payments domain.

Manages a payment session with:
  - BGI risk gate before every transactional call
  - Dispute evidence auto-generation on webhook events
  - Session-level audit log (every assessment, every decision)

In a multi-domain deployment, this would sit alongside
TridentCoordinator in the Swar voice pipeline so a customer-service
agent can answer "why was my payment declined" using the audit log.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from bgi_trident.mcp.protocol import MCPServer, MCPToolResult

logger = logging.getLogger(__name__)


@dataclass
class PaymentSession:
    """Session state for a payment risk assessment flow."""

    session_id: str
    pending_payment_id: str | None = None
    pending_merchant_id: str | None = None
    pending_payer_id: str | None = None
    pending_amount: float = 0.0

    # Audit log: every BGI assessment in this session
    audit_log: list[dict] = field(default_factory=list)

    # Decisions cache (payment_id -> decision dict)
    decisions: dict[str, dict] = field(default_factory=dict)


@dataclass
class RiskExecutionResult:
    """Result from a risk-gated payment operation."""

    operation: str  # "capture_payment", "create_payment_link", etc.
    decision: str  # ALLOW / REVIEW / BLOCK
    ensemble_score: float
    razorpay_result: MCPToolResult | None
    bgi_assessment: dict
    blocked: bool
    summary: str


class PaymentRiskCoordinator:
    """Orchestrates risk-gated Razorpay payment operations.

    Mirrors TridentCoordinator but for the payments domain.
    All transactional calls go through BGI Trident before execution.
    """

    def __init__(
        self,
        razorpay_server: MCPServer,
        bgi_server: MCPServer,
    ) -> None:
        self._razorpay = razorpay_server
        self._bgi = bgi_server
        self._session: PaymentSession | None = None

    async def start_session(self, session_id: str) -> PaymentSession:
        self._session = PaymentSession(session_id=session_id)
        await self._razorpay.connect()
        await self._bgi.connect()
        logger.info("PaymentRiskCoordinator session started: %s", session_id)
        return self._session

    async def create_payment_link(
        self,
        merchant_id: str,
        payer_id: str,
        amount: float,
        description: str = "",
    ) -> RiskExecutionResult:
        """Create a Razorpay payment link -- BGI-gated."""
        from bgi_trident.agents.razorpay import RazorpayFraudAgent

        agent = RazorpayFraudAgent(self._razorpay, self._bgi)
        result = await agent.create_payment_link(
            merchant_id=merchant_id,
            payer_id=payer_id,
            amount=amount,
            description=description,
        )

        bgi_data = result.data.get("bgi_decision", {})
        decision = bgi_data.get("decision", "ALLOW" if result.success else "BLOCK")
        score = bgi_data.get("ensemble_score", 0.0)
        blocked = not result.success and bool(bgi_data)

        self._log_audit("create_payment_link", merchant_id, payer_id, amount, bgi_data)

        return RiskExecutionResult(
            operation="create_payment_link",
            decision=decision,
            ensemble_score=score,
            razorpay_result=result if result.success else None,
            bgi_assessment=bgi_data,
            blocked=blocked,
            summary=(
                f"BLOCKED by BGI Trident (score={score:.3f})" if blocked else f"Payment link created (BGI: {decision}, score={score:.3f})"
            ),
        )

    async def handle_dispute_webhook(
        self,
        payment_id: str,
        dispute_id: str,
        merchant_id: str,
        payer_id: str,
        amount: float,
        reason: str = "customer_dispute",
    ) -> RiskExecutionResult:
        """Handle payment.dispute.created webhook -- auto-generate evidence."""
        from bgi_trident.agents.razorpay import RazorpayFraudAgent

        agent = RazorpayFraudAgent(self._razorpay, self._bgi)
        result = await agent.generate_dispute_evidence(
            payment_id=payment_id,
            dispute_id=dispute_id,
            merchant_id=merchant_id,
            payer_id=payer_id,
            amount=amount,
            reason=reason,
        )
        return RiskExecutionResult(
            operation="generate_dispute_evidence",
            decision="ALLOW",
            ensemble_score=0.0,
            razorpay_result=result,
            bgi_assessment=result.data.get("bgi_evidence", {}),
            blocked=False,
            summary=(f"Dispute evidence generated. Strength: {result.data.get('bgi_evidence', {}).get('evidence_strength', 'N/A')}"),
        )

    async def run_merchant_ring_check(
        self,
        merchant_id: str,
    ) -> RiskExecutionResult:
        """Run BGI ring analysis -- read-only, no Razorpay call."""
        result = await self._bgi.call_tool(
            "detect_merchant_ring",
            {
                "merchant_id": merchant_id,
                "min_shared_payers": 3,
            },
        )
        return RiskExecutionResult(
            operation="detect_merchant_ring",
            decision="REVIEW" if result.success and result.data.get("ring_partners") else "ALLOW",
            ensemble_score=0.0,
            razorpay_result=None,
            bgi_assessment=result.data,
            blocked=False,
            summary=result.data.get("summary", "Ring analysis complete"),
        )

    def _log_audit(
        self,
        operation: str,
        merchant_id: str,
        payer_id: str,
        amount: float,
        bgi_data: dict,
    ) -> None:
        if self._session is None:
            return
        entry = {
            "operation": operation,
            "merchant_id": merchant_id,
            "payer_id": payer_id,
            "amount": amount,
            "decision": bgi_data.get("decision", "N/A"),
            "score": bgi_data.get("ensemble_score", 0.0),
        }
        self._session.audit_log.append(entry)
        if bgi_data.get("payment_id"):
            self._session.decisions[bgi_data["payment_id"]] = bgi_data
        logger.info("Audit: %s", entry)

    async def close(self) -> None:
        await self._razorpay.close()
        await self._bgi.close()
        self._session = None
