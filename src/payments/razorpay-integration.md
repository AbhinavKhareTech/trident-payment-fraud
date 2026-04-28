# Razorpay Integration: BGI Trident Payment Fraud Intelligence

## Overview

This document describes the Razorpay payments domain added to the BGI Trident platform.

The consumption graph (Swiggy: food, instamart, dineout) and the payment fraud graph (Razorpay)
run on the same BGI Trident engine. Same architecture. Different node types, different edge types,
different fraud signals. That is the platform story.

## What was added

| Component | File | Mirrors |
|---|---|---|
| Payment graph schema | `graph/schema_payments.py` | `graph/schema.py` |
| Payment graph builder | `graph/payment_builder.py` | `graph/builder.py` |
| XGBoost payment features | `graph/xgboost/payment_features.py` | `graph/xgboost/features.py` |
| Razorpay MCP server (live) | `mcp/razorpay_server.py` | `mcp/food_server.py` |
| BGI Risk MCP server | `mcp/bgi_risk_server.py` | (new -- exposes 3 fraud tools) |
| BGI Risk engine | `mcp/bgi_risk_engine.py` | (scoring logic) |
| Mock Razorpay MCP | `mcp/mock/razorpay_mock.py` | `mcp/mock/food_mock.py` |
| Razorpay fraud agent | `agents/razorpay.py` | `agents/food.py` |
| Payment risk coordinator | `orchestrator/payment_coordinator.py` | `orchestrator/coordinator.py` |
| Data generator | `data/generate_payment_graph.py` | `data/generate_graph.py` |
| Tests (graph) | `tests/test_payment_graph.py` | `tests/test_graph_builder.py` |
| Tests (agent) | `tests/test_razorpay_agent.py` | `tests/test_agents.py` |
| Tests (fraud) | `tests/test_fraud_detection.py` | `tests/test_ensemble.py` |
| Demo scenario (BLOCK) | `demo/scenarios/razorpay_ring_b.json` | `demo/scenarios/thursday_biryani.json` |
| Demo scenario (ALLOW) | `demo/scenarios/razorpay_clean_allow.json` | |

## Graph schema comparison

| Consumption graph (Swiggy) | Payment fraud graph (Razorpay) |
|---|---|
| USER node | PAYER node |
| RESTAURANT node | MERCHANT node |
| PRODUCT node | DEVICE node |
| VENUE node | BANK_ACCOUNT node |
| TIMESLOT node | IP_ADDRESS node |
| ORDERED_FROM edge | PAID_TO edge |
| OFTEN_PAIRED (cross-domain) | SHARES_BANK (fraud-specific) |
| FOLLOWED_BY_DINING (cross-domain) | RING_PARTNER (derived) |

## Three tools

The `BGIRiskMCPServer` exposes three tools, all implementing the `MCPServer` ABC:

### `assess_payment_risk`

Called before every `capture_payment` or `create_payment_link`.

```python
result = await bgi_server.call_tool("assess_payment_risk", {
    "payment_id":  "pay_P3aB...",
    "merchant_id": "mrc_00005",
    "payer_id":    "pay_00025",
    "amount":      45000.0,
})
# result.data: {"decision": "BLOCK", "ensemble_score": 0.588, ...}
```

### `detect_merchant_ring`

Called during merchant onboarding or on refund spike.

```python
result = await bgi_server.call_tool("detect_merchant_ring", {
    "merchant_id": "mrc_00005",
    "min_shared_payers": 3,
})
```

### `generate_dispute_evidence`

Called on `payment.dispute.created` webhook.

```python
result = await bgi_server.call_tool("generate_dispute_evidence", {
    "payment_id":  "pay_001",
    "dispute_id":  "disp_001",
    "merchant_id": "mrc_00001",
    "payer_id":    "pay_00005",
    "amount":      12500.0,
    "reason":      "customer_dispute",
})
```

## Claude Desktop configuration

```json
{
  "mcpServers": {
    "razorpay": {
      "url": "https://mcp.razorpay.com/mcp",
      "auth": {"type": "basic", "key_id": "YOUR_KEY_ID", "key_secret": "YOUR_KEY_SECRET"}
    },
    "bgi-trident": {
      "command": "python",
      "args": ["-m", "bgi_trident.mcp.bgi_risk_server"],
      "cwd": "/path/to/bgi-razorpay-proper"
    }
  }
}
```

## The 30-second demo

```
User: Create a payment link for INR 50,000 to merchant mrc_00005

Claude:
  1. Calls BGI assess_payment_risk(merchant=mrc_00005, amount=50000)
  2. BGI returns BLOCK (score=0.588):
       SHARED_BANK_ACCOUNT: shares bank with 4 merchants
       MERCHANT_RING_HIGH: 3 HIGH-strength ring partners
  3. Claude refuses to call Razorpay.
  4. Explains: "This merchant is part of a fraud ring. Payment link not created."
```

## Comparison to Swiggy consumption graph

The Swiggy graph proves BGI can model entity relationships in a consumer marketplace.
The Razorpay graph proves BGI can model fraud relationships in a payment network.

Both use the same:
- `MCPServer` ABC for tool access
- `BaseAgent` pattern for domain agents
- `TridentEnsemble` for three-prong scoring
- Coordinator pattern for orchestration

The platform thesis: BGI is domain-agnostic. The graph schema changes.
The prong architecture does not.
