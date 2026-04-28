# Trident Payment Fraud

**Real-time graph-intelligence fraud prevention for Razorpay.**  
Part of the BGI Trident platform — the same engine that powers the [Swiggy consumption graph](https://github.com/AbhinavKhareTech/trident-consumption-graph), now applied to India's payment stack.

![Razorpay](https://img.shields.io/badge/Payments-Razorpay-orange)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![BGI Trident](https://img.shields.io/badge/Platform-BGI_Trident-8A2BE2)
[![Tests](https://img.shields.io/badge/Tests-19%20passing-brightgreen)]()
[![GitHub](https://img.shields.io/badge/GitHub-AbhinavKhareTech/trident--payment--fraud-blue)](https://github.com/AbhinavKhareTech/trident-payment-fraud)

---

## What this is

Razorpay's MCP server gives AI agents the ability to act on payments: create orders, capture payments, issue refunds. What it does not have is a risk layer that reasons over the graph of relationships before those actions execute.

This repo adds that layer.

BGI Trident models every transaction, payer, merchant, device, and bank account as a node in a live heterogeneous graph. Before any money moves, an agent calls three scoring prongs against that graph and receives a decision: ALLOW, REVIEW, or BLOCK. The decision comes with a full explanation traceable to specific graph signals — which ring partners share a bank account, which device appears across multiple payer identities, which refund rate indicates cycling.

The agent then calls Razorpay. Or it does not.

---

## The 30-second demo

Open Claude Desktop with both MCP servers connected. Say:

> "Create a payment link for INR 50,000 to merchant mrc_00005."

Claude calls `assess_payment_risk` from BGI Trident before calling Razorpay.

BGI returns:

```
Decision  : BLOCK
Score     : 0.588

Prong 2 - Graph Intelligence Signals:
  [G] SHARED_BANK_ACCOUNT: merchant mrc_00005 shares bank with 4 other merchants
  [G] MERCHANT_RING_HIGH: 3 HIGH-strength ring partners detected

Ring Partners (3 HIGH-strength):
  - mrc_00006  |  shared_payers=7  |  shared_bank=True
  - mrc_00008  |  shared_payers=5  |  shared_bank=True
  - mrc_00009  |  shared_payers=6  |  shared_bank=True
```

Claude refuses to create the payment link and explains why. No rupee moves.

---

## Architecture

```
AI Agent (Claude Desktop)
    |
    |-- BGI Trident MCP  [THINK before acting]
    |       assess_payment_risk
    |       detect_merchant_ring
    |       generate_dispute_evidence
    |               |
    |               +-- Prong 1: XGBoost (velocity, amount anomaly, failed rate)
    |               +-- Prong 2: Graph-native (shared bank, device mule, ring detection)
    |               +-- Prong 3: Ensemble decision (ALLOW / REVIEW / BLOCK)
    |
    +-- Razorpay MCP  [ACT only after BGI clears]
            create_payment_link | capture_payment | create_refund | ...
```

---

## Three tools

### `assess_payment_risk`

Called before every `capture_payment` or `create_payment_link`. Runs all three prongs and returns a decision with full scoring breakdown and subgraph context.

```python
from bgi_trident.mcp.bgi_risk_engine import PaymentRiskEngine

engine = PaymentRiskEngine(data_dir="src/data")
engine.load()

result = engine.assess_payment_risk(
    payment_id  = "pay_P3aB",
    merchant_id = "mrc_00005",
    payer_id    = "pay_00025",
    amount      = 45000.0,
)
# {"decision": "BLOCK", "ensemble_score": 0.588, "graph_signals": [...], "rings_detected": [...]}
```

### `detect_merchant_ring`

Deep ring analysis for merchant onboarding or refund spike investigation. Detects shared settlement bank accounts, coordinated payer pools, and refund cycling patterns.

### `generate_dispute_evidence`

Called on `payment.dispute.created` webhook. Pulls the transaction subgraph, device fingerprint chain, and behavioural timeline. Returns a structured evidence package ready for Razorpay's dispute API.

---

## Package structure

The payments domain extends the `bgi_trident` package. Every new file mirrors an existing Swiggy-domain file:

```
src/bgi_trident/
    graph/
        schema_payments.py          # PaymentNodeType, PaymentEdgeType, EDGE_REGISTRY
        payment_builder.py          # PaymentGraphBuilder (mirrors builder.py)
        xgboost/
            payment_features.py     # PaymentFraudFeatureExtractor (mirrors features.py)
    mcp/
        razorpay_server.py          # RazorpayMCPServer (implements MCPServer ABC)
        bgi_risk_server.py          # BGIRiskMCPServer (three fraud tools over MCP)
        bgi_risk_engine.py          # PaymentRiskEngine (three-prong scoring logic)
        mock/
            razorpay_mock.py        # MockRazorpayMCP (mirrors food_mock.py)
    agents/
        razorpay.py                 # RazorpayFraudAgent (extends BaseAgent)
    orchestrator/
        payment_coordinator.py      # PaymentRiskCoordinator (mirrors coordinator.py)
src/data/
    generate_payment_graph.py       # Synthetic dataset with 3 fraud rings
tests/
    test_payment_graph.py           # 8 graph construction tests
    test_razorpay_agent.py          # 7 agent tests (mock BGI + mock Razorpay)
    test_fraud_detection.py         # 4 fraud scoring tests
demo/scenarios/
    razorpay_ring_b.json            # BLOCK scenario
    razorpay_clean_allow.json       # ALLOW scenario (precision test)
docs/
    razorpay-integration.md
```

---

## Fraud rings in the synthetic dataset

The data generator injects three rings into a 2,175-transaction baseline (8% fraud rate):

| Ring | Pattern | Scale |
|---|---|---|
| Ring A | Refund cycling | 3 merchants, 15 payers, 40 purchase-refund cycles |
| Ring B | Shared settlement bank | 5 merchants, 1 shared bank account |
| Ring C | Card testing burst | 1 mule payer, 35 micro-transactions in 36-second windows |

---

## Graph schema

| Consumption graph (Swiggy) | Payment fraud graph (Razorpay) |
|---|---|
| USER | PAYER |
| RESTAURANT | MERCHANT |
| PRODUCT | DEVICE |
| VENUE | BANK_ACCOUNT |
| TIMESLOT | IP_ADDRESS |
| ORDERED_FROM | PAID_TO |
| OFTEN_PAIRED (cross-domain) | SHARES_BANK (fraud signal) |
| FOLLOWED_BY_DINING (cross-domain) | RING_PARTNER (derived) |

Same node-edge architecture. Different domain. That is the platform thesis.

---

## Setup

```bash
pip install -e .

# Generate synthetic payment graph (2,175 transactions, 3 fraud rings)
python src/data/generate_payment_graph.py

# Run all 19 new tests
pytest tests/test_payment_graph.py tests/test_razorpay_agent.py tests/test_fraud_detection.py -v
```

## Claude Desktop

Add both servers to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "razorpay": {
      "url": "https://mcp.razorpay.com/mcp",
      "env": {
        "RAZORPAY_KEY_ID": "rzp_test_...",
        "RAZORPAY_KEY_SECRET": "..."
      }
    },
    "bgi-trident": {
      "command": "python",
      "args": ["-m", "bgi_trident.mcp.bgi_risk_server"],
      "cwd": "/path/to/trident-payment-fraud"
    }
  }
}
```

---

## Related

[trident-consumption-graph](https://github.com/AbhinavKhareTech/trident-consumption-graph) - BGI Trident applied to Swiggy's food, instamart, and dineout domains. Autonomous multi-agent ordering with cross-domain graph signals.

---

## Author

**Abhinav Khare** — Cofounder and CTO, [AhinsaAI](https://ahinsaai.com)

20+ years in payments infrastructure, fraud and risk systems, and voice AI for BFSI. Two exits (~$15M each). Built and scaled GCCs from 0 to 500 FTE. ETH Zurich M.S. Engineering, London Business School MBA. 12 active board seats across fintech, banking, and defence. Based in Bangalore.

[LinkedIn](https://linkedin.com/in/abhinavkhare) · [GitHub](https://github.com/AbhinavKhareTech)

---

## License

MIT
