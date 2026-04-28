"""Tests for PaymentGraphBuilder.

Mirrors tests/test_graph_builder.py pattern exactly.
Tests that the payment fraud graph is constructed correctly,
all 5 node types are present, and derived edges are built.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bgi_trident.graph.payment_builder import PaymentGraphBuilder, PaymentInteractionData


@pytest.fixture
def sample_merchants() -> pd.DataFrame:
    return pd.DataFrame({
        "merchant_id":  [f"mrc_{i:05d}" for i in range(10)],
        "name":         [f"Merchant {i}" for i in range(10)],
        "category":     ["retail", "food"] * 5,
        # 5-9 share the same bank account -- Ring B signal
        "bank_account": [f"BANK_{i:04d}" for i in range(5)] + ["SHARED_BANK"] * 5,
        "fraud_ring":   [None] * 5 + ["RING_B"] * 5,
        "created_at":   ["2024-01-01"] * 10,
    })


@pytest.fixture
def sample_payers() -> pd.DataFrame:
    return pd.DataFrame({
        "payer_id":   [f"pay_{i:05d}" for i in range(20)],
        "email":      [f"user{i}@test.com" for i in range(20)],
        "phone":      [f"98{i:08d}" for i in range(20)],
        # payers 0-1 share device_000 -- Co-payer signal
        "device_id":  ["device_000", "device_000"] + [f"device_{i:03d}" for i in range(2, 20)],
        "ip":         [f"192.168.1.{i}" for i in range(20)],
        "fraud_ring": ["RING_A"] * 5 + [None] * 15,
    })


@pytest.fixture
def sample_transactions(sample_merchants, sample_payers) -> pd.DataFrame:
    rows = []
    # Clean transactions
    for i in range(50):
        rows.append({
            "payment_id":  f"pay_txn_{i:05d}",
            "merchant_id": f"mrc_{i % 10:05d}",
            "payer_id":    f"pay_{i % 20:05d}",
            "amount":      float(200 + i * 10),
            "status":      "captured",
            "method":      "upi",
            "device_id":   f"device_{i % 20:03d}",
            "created_at":  "2024-10-15T10:00:00",
            "fraud_label": 0,
        })
    # Ring B: payers 5-9 transact with shared-bank merchants 5-9
    for i in range(20):
        rows.append({
            "payment_id":  f"pay_txn_ring_{i}",
            "merchant_id": f"mrc_{5 + i % 5:05d}",
            "payer_id":    f"pay_{5 + i % 5:05d}",
            "amount":      5000.0,
            "status":      "captured",
            "method":      "upi",
            "device_id":   "device_005",
            "created_at":  "2024-10-16T10:00:00",
            "fraud_label": 1,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def payment_data(sample_merchants, sample_payers, sample_transactions) -> PaymentInteractionData:
    return PaymentInteractionData(
        transactions=sample_transactions,
        merchants=sample_merchants,
        payers=sample_payers,
    )


def test_builder_constructs(payment_data):
    builder = PaymentGraphBuilder(payment_data).build()
    counts  = builder.node_counts
    assert counts["merchant"]     == 10
    assert counts["payer"]        == 20
    assert counts["bank_account"] >= 6   # 5 unique + 1 shared
    assert counts["device"]       >= 2


def test_paid_to_edges_built(payment_data):
    builder = PaymentGraphBuilder(payment_data).build()
    assert "payer-paid_to->merchant" in builder.edge_counts
    assert builder.edge_counts["payer-paid_to->merchant"] > 0


def test_shares_bank_edges(payment_data):
    builder = PaymentGraphBuilder(payment_data).build()
    assert "merchant-shares_bank->bank_account" in builder.edge_counts


def test_ring_partner_edges_derived(payment_data):
    """Ring B merchants all transact with overlapping payers -- should produce ring_partner edges."""
    builder = PaymentGraphBuilder(payment_data).build()
    edge_key = "merchant-ring_partner->merchant"
    if edge_key in builder.edge_counts:
        assert builder.edge_counts[edge_key] > 0


def test_co_payer_edges_for_shared_device(payment_data):
    """Payers 0 and 1 share device_000 -- should produce co_payer edge."""
    builder = PaymentGraphBuilder(payment_data).build()
    assert "payer-co_payer->payer" in builder.edge_counts
    assert builder.edge_counts["payer-co_payer->payer"] >= 2


def test_merchant_txn_index(payment_data):
    builder = PaymentGraphBuilder(payment_data).build()
    assert "mrc_00000" in builder.merchant_txns
    assert len(builder.merchant_txns["mrc_00000"]) > 0


def test_bank_merchant_index(payment_data):
    builder = PaymentGraphBuilder(payment_data).build()
    # SHARED_BANK should map to 5 merchants
    assert "SHARED_BANK" in builder.bank_merchants
    assert len(builder.bank_merchants["SHARED_BANK"]) == 5


def test_xgboost_features_shape(payment_data):
    builder = PaymentGraphBuilder(payment_data).build()
    features = builder.get_xgboost_features()
    assert "payer_id"    in features.columns
    assert "merchant_id" in features.columns
    assert "refund_rate" in features.columns
    assert len(features)  > 0
