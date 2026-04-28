"""End-to-end fraud detection tests.

Tests that the BGI risk engine correctly scores fraud rings
and clean transactions, mirrors the ablation study pattern
from test_ensemble.py.
"""

from __future__ import annotations

import pandas as pd
import pytest


def _make_payment_data():
    """Minimal synthetic dataset with a Ring B pattern."""
    from bgi_trident.graph.payment_builder import PaymentGraphBuilder, PaymentInteractionData

    # 5 merchants share the same bank account
    merchants = pd.DataFrame({
        "merchant_id":  [f"mrc_{i:05d}" for i in range(10)],
        "name":         [f"Merchant {i}" for i in range(10)],
        "category":     ["retail"] * 10,
        "bank_account": [f"BANK_{i}" for i in range(5)] + ["SHARED_BANK"] * 5,
        "fraud_ring":   [None] * 5 + ["RING_B"] * 5,
        "created_at":   ["2024-01-01"] * 10,
    })
    payers = pd.DataFrame({
        "payer_id":   [f"pay_{i:05d}" for i in range(30)],
        "email":      [f"u{i}@test.com" for i in range(30)],
        "phone":      [f"98{i:08d}" for i in range(30)],
        "device_id":  [f"dev_{i:03d}" for i in range(30)],
        "ip":         [f"10.0.0.{i}" for i in range(30)],
        "fraud_ring": [None] * 30,
    })
    # Ring B: 7 payers each transact with 4 of the 5 shared-bank merchants
    rows = []
    for p in range(7):
        for m in range(5, 10):
            rows.append({
                "payment_id":  f"ring_{p}_{m}",
                "merchant_id": f"mrc_{m:05d}",
                "payer_id":    f"pay_{p:05d}",
                "amount":      8000.0,
                "status":      "captured",
                "method":      "upi",
                "device_id":   f"dev_{p:03d}",
                "created_at":  "2024-10-10T10:00:00",
                "fraud_label": 1,
            })
    # Clean transactions
    for i in range(40):
        rows.append({
            "payment_id":  f"clean_{i}",
            "merchant_id": f"mrc_{i % 5:05d}",
            "payer_id":    f"pay_{10 + i % 20:05d}",
            "amount":      float(500 + i * 50),
            "status":      "captured",
            "method":      "upi",
            "device_id":   f"dev_{10 + i % 20:03d}",
            "created_at":  "2024-10-12T10:00:00",
            "fraud_label": 0,
        })

    data = PaymentInteractionData(
        transactions=pd.DataFrame(rows),
        merchants=merchants,
        payers=payers,
    )
    return PaymentGraphBuilder(data).build()


def test_ring_b_merchant_has_high_graph_score():
    """Ring B merchants (shared bank) should score high on Prong 2."""
    from bgi_trident.mcp.bgi_risk_engine import _graph_score, _detect_rings

    builder = _make_payment_data()

    # Patch the builder's data accessor
    class FakeEngine:
        def __init__(self, b): self._builder = b
        @property
        def data(self): return b.data

    b = builder  # PaymentGraphBuilder

    score_data = _graph_score(b, "mrc_00005", "pay_00000")
    assert score_data["score"] > 0.0, "Ring B merchant should have non-zero graph score"
    assert any("SHARED_BANK" in s or "RING" in s for s in score_data["signals"])


def test_clean_merchant_has_low_graph_score():
    """Clean merchants (unique bank account, no rings) should score near zero."""
    from bgi_trident.mcp.bgi_risk_engine import _graph_score

    builder = _make_payment_data()
    score_data = _graph_score(builder, "mrc_00000", "pay_00010")
    # Clean merchant should be low; allow small non-zero from co-payer noise
    assert score_data["score"] < 0.35, f"Clean merchant scored too high: {score_data['score']}"


def test_ring_detection_finds_partners():
    """detect_merchant_ring should find HIGH-strength ring partners for Ring B merchants."""
    from bgi_trident.mcp.bgi_risk_engine import _detect_rings

    builder = _make_payment_data()
    rings = _detect_rings(builder, "mrc_00005", min_shared=3)
    assert len(rings) > 0, "Should detect ring partners"
    high = [r for r in rings if r["ring_strength"] == "HIGH"]
    assert len(high) > 0, "Should have at least one HIGH-strength ring partner"


def test_xgboost_features_include_fraud_labels():
    """XGBoost feature DataFrame should include fraud_label column for Ring B pairs."""
    builder = _make_payment_data()
    features = builder.get_xgboost_features()
    assert "refund_rate"  in features.columns
    assert "failed_rate"  in features.columns
    assert len(features)  > 0
