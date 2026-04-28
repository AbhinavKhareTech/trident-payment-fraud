"""Tests for RazorpayFraudAgent.

Mirrors tests/test_agents.py pattern. Uses mock servers so
no live Razorpay API keys are needed.
"""

from __future__ import annotations

import pytest
from bgi_trident.agents.razorpay import RazorpayFraudAgent

from bgi_trident.mcp.protocol import MCPServer, MCPToolResult

# ── Minimal mock servers for testing ─────────────────────────────────────────


class MockBGIAllow(MCPServer):
    """BGI server that always returns ALLOW."""

    async def connect(self):
        pass

    async def close(self):
        pass

    def available_tools(self):
        return ["assess_payment_risk", "detect_merchant_ring", "generate_dispute_evidence"]

    async def call_tool(self, tool_name, params):
        if tool_name == "assess_payment_risk":
            return MCPToolResult(
                tool_name=tool_name,
                success=True,
                data={
                    "decision": "ALLOW",
                    "ensemble_score": 0.15,
                    "explanation": "No signals detected.",
                    "graph_signals": [],
                    "rings_detected": [],
                },
            )
        if tool_name == "detect_merchant_ring":
            return MCPToolResult(
                tool_name=tool_name,
                success=True,
                data={
                    "ring_partners": [],
                    "summary": "No ring detected.",
                },
            )
        if tool_name == "generate_dispute_evidence":
            return MCPToolResult(
                tool_name=tool_name,
                success=True,
                data={
                    "evidence_strength": "MODERATE",
                    "narrative": "Evidence generated.",
                    "device_fingerprint_chain": [{"device_id": "dev_001", "ip_address": "1.2.3.4"}],
                },
            )
        return MCPToolResult(tool_name=tool_name, success=False, data={}, error="unknown")


class MockBGIBlock(MCPServer):
    """BGI server that always returns BLOCK."""

    async def connect(self):
        pass

    async def close(self):
        pass

    def available_tools(self):
        return ["assess_payment_risk"]

    async def call_tool(self, tool_name, params):
        return MCPToolResult(
            tool_name=tool_name,
            success=True,
            data={
                "decision": "BLOCK",
                "ensemble_score": 0.82,
                "explanation": "Ring B detected.",
                "graph_signals": ["SHARED_BANK_ACCOUNT: merchant shares bank with 4 others"],
                "rings_detected": [{"ring_merchant": "mrc_00006", "shared_payer_count": 6, "ring_strength": "HIGH"}],
            },
        )


class MockRazorpay(MCPServer):
    """Minimal Razorpay mock."""

    def __init__(self):
        self.calls = []

    async def connect(self):
        pass

    async def close(self):
        pass

    def available_tools(self):
        return ["create_payment_link", "capture_payment", "fetch_dispute"]

    async def call_tool(self, tool_name, params):
        self.calls.append((tool_name, params))
        return MCPToolResult(
            tool_name=tool_name, success=True, data={"id": "plink_TEST", "status": "created", "short_url": "https://rzp.io/i/TEST"}
        )


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def razorpay_mock():
    return MockRazorpay()


@pytest.mark.asyncio
async def test_create_payment_link_allow(razorpay_mock):
    agent = RazorpayFraudAgent(razorpay_mock, MockBGIAllow())
    result = await agent.create_payment_link(
        merchant_id="mrc_00050",
        payer_id="pay_00300",
        amount=1499.0,
        description="Test payment",
    )
    assert result.success, f"Expected success but got: {result.error}"
    # Razorpay was called
    assert any(c[0] == "create_payment_link" for c in razorpay_mock.calls)


@pytest.mark.asyncio
async def test_create_payment_link_blocked(razorpay_mock):
    agent = RazorpayFraudAgent(razorpay_mock, MockBGIBlock())
    result = await agent.create_payment_link(
        merchant_id="mrc_00005",
        payer_id="pay_00025",
        amount=45000.0,
        description="Ring B merchant",
    )
    assert not result.success, "Expected BLOCK to prevent success"
    assert "BLOCK" in (result.error or "")
    # Razorpay was NOT called
    assert not any(c[0] == "create_payment_link" for c in razorpay_mock.calls)


@pytest.mark.asyncio
async def test_execute_order_allow(razorpay_mock):
    agent = RazorpayFraudAgent(razorpay_mock, MockBGIAllow())
    result = await agent.execute_order(
        payment_id="pay_001",
        merchant_id="mrc_00050",
        payer_id="pay_00300",
        amount=1499.0,
    )
    assert result.success


@pytest.mark.asyncio
async def test_execute_order_blocked(razorpay_mock):
    agent = RazorpayFraudAgent(razorpay_mock, MockBGIBlock())
    result = await agent.execute_order(
        payment_id="pay_001",
        merchant_id="mrc_00005",
        payer_id="pay_00025",
        amount=45000.0,
    )
    assert not result.success
    assert "BLOCKED" in (result.error or "")


@pytest.mark.asyncio
async def test_generate_dispute_evidence(razorpay_mock):
    agent = RazorpayFraudAgent(razorpay_mock, MockBGIAllow())
    result = await agent.generate_dispute_evidence(
        payment_id="pay_001",
        dispute_id="disp_001",
        merchant_id="mrc_00001",
        payer_id="pay_00005",
        amount=12500.0,
        reason="customer_dispute",
    )
    assert result.success
    assert "bgi_evidence" in result.data
    assert "razorpay_dispute" in result.data
    assert result.data.get("ready_to_submit") in (True, False)


@pytest.mark.asyncio
async def test_detect_merchant_ring_delegates_to_bgi(razorpay_mock):
    agent = RazorpayFraudAgent(razorpay_mock, MockBGIAllow())
    result = await agent.detect_merchant_ring("mrc_00007")
    assert result.success
    # BGI was called, Razorpay was NOT
    assert not razorpay_mock.calls


@pytest.mark.asyncio
async def test_search_is_read_only(razorpay_mock):
    agent = RazorpayFraudAgent(razorpay_mock, MockBGIAllow())
    result = await agent.search("list transactions")
    # list_payments called directly, no BGI gate
    assert any(c[0] == "list_payments" for c in razorpay_mock.calls)
