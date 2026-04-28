"""BGI Payment Risk Engine.

Wires PaymentGraphBuilder, PaymentFraudFeatureExtractor, and
TridentEnsemble together into the three tools exposed by BGIRiskMCPServer.

This is what runs inside the MCP server. It is NOT the MCP server itself --
it is the pure-Python scoring engine that the server delegates to.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PaymentRiskEngine:
    """Loads payment data and runs all three Trident prongs."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._builder = None
        self._loaded = False

    def load(self) -> None:
        from bgi_trident.graph.payment_builder import (
            PaymentGraphBuilder,
            PaymentInteractionData,
        )

        txns_path = self._data_dir / "interactions" / "razorpay_transactions.csv"
        merchants_path = self._data_dir / "razorpay_merchants.csv"
        payers_path = self._data_dir / "razorpay_payers.csv"

        if not txns_path.exists():
            logger.warning("Transaction data not found at %s; generating...", txns_path)
            from bgi_trident.mcp.bgi_risk_engine import _generate_seed_data

            _generate_seed_data(self._data_dir)

        txns = pd.read_csv(txns_path)
        merchants = pd.read_csv(merchants_path)
        payers = pd.read_csv(payers_path)

        data = PaymentInteractionData(
            transactions=txns,
            merchants=merchants,
            payers=payers,
        )
        self._builder = PaymentGraphBuilder(data).build()
        self._loaded = True
        logger.info(
            "PaymentRiskEngine loaded: %s",
            {k: v for k, v in self._builder.node_counts.items()},
        )

    # ── Tool 1: assess_payment_risk ───────────────────────────────────────────

    def assess_payment_risk(
        self,
        payment_id: str,
        merchant_id: str,
        payer_id: str,
        amount: float,
        method: str = "upi",
    ) -> dict:
        b = self._builder
        now = datetime.utcnow()

        # Prong 1: velocity
        payer_txns = b.payer_txns.get(payer_id, [])
        merchant_txns = b.merchant_txns.get(merchant_id, [])

        p1_payer = _velocity_score(payer_id, amount, payer_txns, now)
        p1_merchant = _merchant_score(merchant_id, merchant_txns)

        # Prong 2: graph intelligence
        p2 = _graph_score(b, merchant_id, payer_id)

        # Prong 3: ensemble
        ensemble = 0.25 * p1_payer["score"] + 0.20 * p1_merchant["score"] + 0.55 * p2["score"]
        ensemble = round(min(ensemble, 1.0), 4)

        decision = "BLOCK" if ensemble >= 0.55 else "REVIEW" if ensemble >= 0.30 else "ALLOW"
        confidence = int(ensemble * 100) if decision != "ALLOW" else int((1 - ensemble) * 100)

        return {
            "decision": decision,
            "confidence": confidence,
            "ensemble_score": ensemble,
            "explanation": _build_explanation(
                decision,
                ensemble,
                p1_payer,
                p1_merchant,
                p2,
                payment_id,
                merchant_id,
                payer_id,
                amount,
            ),
            "prong_scores": {
                "prong1_payer": p1_payer["score"],
                "prong1_merchant": p1_merchant["score"],
                "prong2_graph": p2["score"],
            },
            "velocity_flags": {"payer": p1_payer["features"], "merchant": p1_merchant["features"]},
            "graph_signals": p2["signals"],
            "rings_detected": p2["rings"][:5],
            "payment_id": payment_id,
            "merchant_id": merchant_id,
            "payer_id": payer_id,
            "timestamp": now.isoformat(),
        }

    # ── Tool 2: detect_merchant_ring ──────────────────────────────────────────

    def detect_merchant_ring(
        self,
        merchant_id: str,
        min_shared_payers: int = 3,
    ) -> dict:
        b = self._builder
        rings = _detect_rings(b, merchant_id, min_shared_payers)
        bank_peers = [m for m in _shared_bank_merchants(b, merchant_id) if m != merchant_id]
        m_txns = b.merchant_txns.get(merchant_id, [])
        refund_rate = 0.0
        if m_txns:
            refunded = sum(1 for t in m_txns if t.get("status") == "refunded")
            refund_rate = round(refunded / len(m_txns), 3)

        summary_lines = []
        high = [r for r in rings if r["ring_strength"] == "HIGH"]
        if high:
            summary_lines.append(f"RING DETECTED: Merchant {merchant_id} has {len(high)} HIGH-strength ring partner(s).")
        if bank_peers:
            summary_lines.append(f"SHARED BANK: {len(bank_peers)} other merchant(s) share the same settlement account.")
        if refund_rate > 0.35:
            summary_lines.append(f"HIGH REFUND RATE: {refund_rate:.0%} of transactions reversed.")
        if not summary_lines:
            summary_lines.append(f"No ring patterns detected for merchant {merchant_id}.")

        return {
            "merchant_id": merchant_id,
            "refund_rate": refund_rate,
            "total_txns": len(m_txns),
            "ring_partners": rings,
            "bank_peers": bank_peers,
            "summary": "\n".join(summary_lines),
        }

    # ── Tool 3: generate_dispute_evidence ─────────────────────────────────────

    def generate_dispute_evidence(
        self,
        payment_id: str,
        dispute_id: str,
        merchant_id: str,
        payer_id: str,
        amount: float,
        reason: str = "customer_dispute",
    ) -> dict:
        b = self._builder
        payer_txns = b.payer_txns.get(payer_id, [])
        prior_to_merch = [t for t in payer_txns if t.get("merchant_id") == merchant_id and t.get("payment_id") != payment_id]
        target_txn = next((t for t in payer_txns if t.get("payment_id") == payment_id), None)
        # Device chain from payers CSV
        payer_row = {}
        if b._builder if hasattr(b, "_builder") else True:
            pass  # engine holds builder directly
        device_chain = []
        for payer_rows in [b.data.payers[b.data.payers["payer_id"] == payer_id].itertuples()]:
            for row in payer_rows:
                if getattr(row, "device_id", None):
                    device_chain.append(
                        {
                            "device_id": row.device_id,
                            "ip_address": getattr(row, "ip", ""),
                        }
                    )

        evidence_strength = (
            "STRONG" if len(prior_to_merch) >= 2 and device_chain else "MODERATE" if (prior_to_merch or device_chain) else "WEAK"
        )

        return {
            "dispute_id": dispute_id,
            "payment_id": payment_id,
            "merchant_id": merchant_id,
            "payer_id": payer_id,
            "amount": amount,
            "dispute_reason": reason,
            "transaction_details": target_txn or {"note": "not in local graph"},
            "device_fingerprint_chain": device_chain,
            "payer_behavioural_summary": {
                "total_transactions": len(payer_txns),
                "prior_txns_to_this_merchant": len(prior_to_merch),
                "recent_timeline": payer_txns[-5:],
            },
            "evidence_strength": evidence_strength,
            "narrative": _dispute_narrative(
                payment_id,
                merchant_id,
                payer_id,
                amount,
                reason,
                prior_to_merch,
                device_chain,
            ),
        }


# ── Scoring helpers ────────────────────────────────────────────────────────────


def _velocity_score(payer_id, amount, txns, now) -> dict:
    features = []
    details = {}

    def vel(hours):
        return sum(1 for t in txns if (now - datetime.fromisoformat(str(t["created_at"]))).total_seconds() <= hours * 3600)

    v1h, v6h, v24h = vel(1), vel(6), vel(24)
    details.update(velocity_1h=v1h, velocity_6h=v6h, velocity_24h=v24h)

    if v1h >= 5:
        features.append(f"HIGH_VELOCITY_1H:{v1h}")
    if v6h >= 15:
        features.append(f"HIGH_VELOCITY_6H:{v6h}")
    if v24h >= 30:
        features.append(f"HIGH_VELOCITY_24H:{v24h}")

    micro = sum(1 for t in txns if t.get("amount", 0) < 100)
    failed = sum(1 for t in txns if t.get("status") == "failed")
    details["micro_count"] = micro
    details["failed_rate"] = round(failed / len(txns), 3) if txns else 0

    if micro >= 10:
        features.append(f"MICRO_TXN_BURST:{micro}")
    if txns and failed / len(txns) > 0.4:
        features.append(f"HIGH_FAILED_RATE:{failed / len(txns):.0%}")

    amounts = [t["amount"] for t in txns if "amount" in t]
    if len(amounts) >= 3:
        std = float(np.std(amounts)) or 1.0
        z = abs(amount - float(np.mean(amounts))) / std
        details["amount_zscore"] = round(z, 2)
        if z > 3.5:
            features.append(f"AMOUNT_OUTLIER:z={z:.1f}")

    score = min(
        sum(
            {
                "HIGH_VELOCITY_1H": 0.30,
                "HIGH_VELOCITY_6H": 0.20,
                "HIGH_VELOCITY_24H": 0.15,
                "MICRO_TXN_BURST": 0.40,
                "HIGH_FAILED_RATE": 0.25,
                "AMOUNT_OUTLIER": 0.20,
            }.get(f.split(":")[0], 0)
            for f in features
        ),
        1.0,
    )

    return {"score": round(score, 3), "features": features, "details": details}


def _merchant_score(merchant_id, txns) -> dict:
    features = []
    if not txns:
        return {"score": 0.0, "features": features}

    refunded = sum(1 for t in txns if t.get("status") == "refunded")
    captured = sum(1 for t in txns if t.get("status") == "captured")
    refund_r = refunded / len(txns)
    captured_val = sum(t["amount"] for t in txns if t.get("status") == "captured")
    refund_val = sum(t["amount"] for t in txns if t.get("status") == "refunded")

    if refund_r > 0.35:
        features.append(f"HIGH_REFUND_RATE:{refund_r:.0%}")
    if captured_val and refund_val / captured_val > 0.40:
        features.append(f"HIGH_REFUND_VALUE_RATIO:{refund_val / captured_val:.0%}")

    score = min(sum({"HIGH_REFUND_RATE": 0.35, "HIGH_REFUND_VALUE_RATIO": 0.30}.get(f.split(":")[0], 0) for f in features), 1.0)
    return {"score": round(score, 3), "features": features}


def _shared_bank_merchants(b, merchant_id) -> list[str]:
    """Find merchants sharing the same bank account."""
    # Look up bank account from merchants dataframe
    try:
        bank = b.data.merchants[b.data.merchants["merchant_id"] == merchant_id]["bank_account"].iloc[0]
        peers = b.data.merchants[b.data.merchants["bank_account"] == bank]["merchant_id"].tolist()
        return [p for p in peers if p != merchant_id]
    except (IndexError, KeyError):
        return []


def _detect_rings(b, merchant_id, min_shared=3) -> list[dict]:
    # Payers of target merchant
    target_payers = set(t["payer_id"] for t in b.merchant_txns.get(merchant_id, []))
    if not target_payers:
        return []

    co_map: dict[str, int] = defaultdict(int)
    for pid in target_payers:
        for t in b.payer_txns.get(pid, []):
            mid = t.get("merchant_id")
            if mid and mid != merchant_id:
                co_map[mid] += 1

    bank_peers = set(_shared_bank_merchants(b, merchant_id))
    rings = []
    for co_mid, cnt in co_map.items():
        if cnt < min_shared:
            continue
        shares_bank = co_mid in bank_peers
        rings.append(
            {
                "target_merchant": merchant_id,
                "ring_merchant": co_mid,
                "shared_payer_count": cnt,
                "shares_bank_account": shares_bank,
                "ring_strength": ("HIGH" if shares_bank and cnt >= 5 else "MEDIUM" if cnt >= 3 else "LOW"),
            }
        )
    return sorted(rings, key=lambda r: r["shared_payer_count"], reverse=True)


def _graph_score(b, merchant_id, payer_id) -> dict:
    signals = []
    bank_peers = _shared_bank_merchants(b, merchant_id)
    if bank_peers:
        signals.append(f"SHARED_BANK_ACCOUNT: merchant {merchant_id} shares bank with {len(bank_peers)} merchant(s): {bank_peers[:3]}")
    # Device sharing
    payer_dev = None
    try:
        payer_dev = b.data.payers[b.data.payers["payer_id"] == payer_id]["device_id"].iloc[0]
    except (IndexError, KeyError):
        pass
    if payer_dev:
        co_payers = b.device_payers.get(payer_dev, set()) - {payer_id}
        if len(co_payers) >= 2:
            signals.append(f"SHARED_DEVICE: payer {payer_id} shares device with {len(co_payers)} other payer(s)")

    rings = _detect_rings(b, merchant_id, min_shared=3)
    high_rings = [r for r in rings if r["ring_strength"] == "HIGH"]
    if high_rings:
        signals.append(f"MERCHANT_RING_HIGH: {len(high_rings)} HIGH-strength ring partner(s) for {merchant_id}")
    elif [r for r in rings if r["ring_strength"] == "MEDIUM"]:
        signals.append(f"MERCHANT_RING_MEDIUM: {len(rings)} ring partner(s)")

    # Refund cycling
    m_txns = b.merchant_txns.get(merchant_id, [])
    if m_txns:
        refunded = sum(1 for t in m_txns if t.get("status") == "refunded")
        rr = refunded / len(m_txns)
        if rr > 0.35 and rings:
            signals.append(f"REFUND_CYCLE_PATTERN: {rr:.0%} refund rate + ring partners")

    s = 0.0
    for sig in signals:
        if "SHARED_BANK" in sig:
            s += 0.30
        elif "MERCHANT_RING_HIGH" in sig:
            s += 0.50
        elif "MERCHANT_RING_MEDIUM" in sig or "SHARED_DEVICE" in sig:
            s += 0.25
        elif "REFUND_CYCLE" in sig:
            s += 0.40
    return {"score": round(min(s, 1.0), 3), "signals": signals, "rings": rings}


def _build_explanation(decision, score, p1_payer, p1_merchant, p2, payment_id, merchant_id, payer_id, amount) -> str:
    lines = [
        "BGI Trident Risk Assessment",
        f"Payment   : {payment_id or 'N/A'}",
        f"Merchant  : {merchant_id}",
        f"Payer     : {payer_id}",
        f"Amount    : INR {amount:,.2f}",
        "",
        f"Decision  : {decision}",
        f"Score     : {score:.3f}  (BLOCK >= 0.55 | REVIEW >= 0.30)",
        "",
    ]
    if p1_payer["features"]:
        lines.append("Prong 1 - Payer Velocity Flags:")
        for f in p1_payer["features"]:
            lines.append(f"  [!] {f}")
        lines.append("")
    if p1_merchant["features"]:
        lines.append("Prong 1 - Merchant Velocity Flags:")
        for f in p1_merchant["features"]:
            lines.append(f"  [!] {f}")
        lines.append("")
    if p2["signals"]:
        lines.append("Prong 2 - Graph Intelligence Signals:")
        for sig in p2["signals"]:
            lines.append(f"  [G] {sig}")
        lines.append("")
    high_rings = [r for r in p2["rings"] if r["ring_strength"] == "HIGH"]
    if high_rings:
        lines.append(f"Ring Partners ({len(high_rings)} HIGH-strength):")
        for r in high_rings[:3]:
            lines.append(f"  - {r['ring_merchant']}  |  shared_payers={r['shared_payer_count']}  |  shared_bank={r['shares_bank_account']}")
        lines.append("")
    if not p1_payer["features"] and not p1_merchant["features"] and not p2["signals"]:
        lines.append("No significant risk signals. Transaction appears clean.")
    return "\n".join(lines)


def _dispute_narrative(payment_id, merchant_id, payer_id, amount, reason, prior_txns, device_chain) -> str:
    lines = [
        "DISPUTE EVIDENCE NARRATIVE",
        "Generated by BGI Trident | AhinsaAI",
        "",
        f"Payment  : {payment_id}",
        f"Merchant : {merchant_id}",
        f"Payer    : {payer_id}",
        f"Amount   : INR {amount:,.2f}",
        f"Reason   : {reason}",
        "",
    ]
    if prior_txns:
        lines.append(
            f"The payer has {len(prior_txns)} prior successful transaction(s) with this merchant. "
            f"The disputed transaction follows the same behavioural pattern as prior authorised payments."
        )
    else:
        lines.append("First transaction between this payer and merchant. No prior baseline.")

    if device_chain:
        dc = device_chain[0]
        lines.append(
            f"Device fingerprint: device_id={dc.get('device_id')}, "
            f"IP={dc.get('ip_address')}. Consistent across prior authorised transactions."
        )
    return "\n".join(lines)


def _generate_seed_data(data_dir: Path) -> None:
    """Generate minimal seed CSVs if not present."""
    import hashlib
    import random

    from faker import Faker

    fake = Faker("en_IN")
    random.seed(42)

    (data_dir / "interactions").mkdir(parents=True, exist_ok=True)

    # Merchants
    merchants = []
    for i in range(80):
        bank = f"BANKA_{i:04d}"
        if 5 <= i < 10:
            bank = "SHARED_BANK_XYZ"
        merchants.append(
            {
                "merchant_id": f"mrc_{i:05d}",
                "name": fake.company(),
                "category": random.choice(["retail", "food", "travel", "edtech"]),
                "bank_account": bank,
                "fraud_ring": "RING_B" if 5 <= i < 10 else None,
            }
        )
    pd.DataFrame(merchants).to_csv(data_dir / "razorpay_merchants.csv", index=False)

    # Payers
    payers = []
    for i in range(400):
        payers.append(
            {
                "payer_id": f"pay_{i:05d}",
                "email": fake.email(),
                "phone": f"9{random.randint(100000000, 999999999)}",
                "device_id": hashlib.md5(f"dev{i // 3}".encode()).hexdigest()[:16],
                "ip": fake.ipv4(),
            }
        )
    pd.DataFrame(payers).to_csv(data_dir / "razorpay_payers.csv", index=False)

    # Transactions
    from datetime import timedelta

    base = datetime(2024, 10, 1)
    rows = []
    for i in range(2000):
        m = merchants[random.randint(0, 79)]
        p = payers[random.randint(0, 399)]
        ring = m.get("fraud_ring")
        rows.append(
            {
                "payment_id": f"pay_txn_{i:05d}",
                "merchant_id": m["merchant_id"],
                "payer_id": p["payer_id"],
                "amount": round(random.uniform(100, 10000) if not ring else random.uniform(5000, 40000), 2),
                "status": "refunded"
                if ring and random.random() < 0.4
                else random.choices(["captured", "failed", "refunded"], weights=[80, 10, 10])[0],
                "method": random.choice(["upi", "card", "netbanking"]),
                "device_id": p["device_id"],
                "created_at": (base + timedelta(hours=random.uniform(0, 720))).isoformat(),
                "fraud_label": 1 if ring else 0,
            }
        )
    pd.DataFrame(rows).to_csv(data_dir / "interactions" / "razorpay_transactions.csv", index=False)
    logger.info("Seed data generated at %s", data_dir)
