"""Graph schema definitions for the BGI Trident payment fraud graph.

Mirrors the consumption graph schema pattern (schema.py) but for the
Razorpay payments domain: five node types, seven edge types.

The fraud-specific cross-domain edges (SHARES_BANK, SHARED_DEVICE)
are the equivalent of OFTEN_PAIRED / FOLLOWED_BY_DINING in the
consumption graph -- the edges a naive single-entity fraud system
cannot represent.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PaymentNodeType(str, Enum):
    """Five node types in the payment fraud graph."""

    MERCHANT     = "merchant"
    PAYER        = "payer"
    DEVICE       = "device"
    BANK_ACCOUNT = "bank_account"
    IP_ADDRESS   = "ip_address"


class PaymentEdgeType(str, Enum):
    """Seven edge types.

    SHARES_BANK and SHARED_DEVICE are the fraud-specific cross-entity
    edges that single-record fraud systems cannot represent.
    """

    PAID_TO       = "paid_to"        # Payer -> Merchant (transaction)
    REFUNDED_BY   = "refunded_by"    # Merchant -> Payer (refund)
    SHARES_DEVICE = "shares_device"  # Payer -> Device
    SAME_IP       = "same_ip"        # Payer -> IPAddress
    SHARES_BANK   = "shares_bank"    # Merchant -> BankAccount
    CO_PAYER      = "co_payer"       # Payer -> Payer (derived: same device)
    RING_PARTNER  = "ring_partner"   # Merchant -> Merchant (derived: shared payers)


# Canonical edge tuples for PyG HeteroData and DGL DGLHeteroGraph
PAYMENT_EDGE_REGISTRY: dict[PaymentEdgeType, tuple[PaymentNodeType, str, PaymentNodeType]] = {
    PaymentEdgeType.PAID_TO:      (PaymentNodeType.PAYER,    "paid_to",       PaymentNodeType.MERCHANT),
    PaymentEdgeType.REFUNDED_BY:  (PaymentNodeType.MERCHANT, "refunded_by",   PaymentNodeType.PAYER),
    PaymentEdgeType.SHARES_DEVICE:(PaymentNodeType.PAYER,    "shares_device", PaymentNodeType.DEVICE),
    PaymentEdgeType.SAME_IP:      (PaymentNodeType.PAYER,    "same_ip",       PaymentNodeType.IP_ADDRESS),
    PaymentEdgeType.SHARES_BANK:  (PaymentNodeType.MERCHANT, "shares_bank",   PaymentNodeType.BANK_ACCOUNT),
    PaymentEdgeType.CO_PAYER:     (PaymentNodeType.PAYER,    "co_payer",      PaymentNodeType.PAYER),
    PaymentEdgeType.RING_PARTNER: (PaymentNodeType.MERCHANT, "ring_partner",  PaymentNodeType.MERCHANT),
}


@dataclass
class PaymentNodeFeatureSpec:
    """Feature vector spec for a payment graph node type."""

    node_type:     PaymentNodeType
    feature_names: list[str]
    embedding_dim: int
    description:   str = ""


@dataclass
class PaymentEdgeWeightSpec:
    """Weight signals on each payment edge type."""

    edge_type:         PaymentEdgeType
    weight_signals:    list[str]
    has_temporal_decay: bool = False
    description:       str = ""


PAYMENT_NODE_FEATURES: dict[PaymentNodeType, PaymentNodeFeatureSpec] = {
    PaymentNodeType.MERCHANT: PaymentNodeFeatureSpec(
        node_type=PaymentNodeType.MERCHANT,
        feature_names=[
            "category_vector",        # Multi-hot (10 merchant categories)
            "refund_rate",            # Float [0, 1]
            "avg_txn_amount",         # INR
            "txn_velocity_24h",       # Count
            "settlement_age_days",    # Float
            "is_new_merchant",        # Boolean (< 30 days)
        ],
        embedding_dim=64,
        description="Merchant attributes and behavioural signals",
    ),
    PaymentNodeType.PAYER: PaymentNodeFeatureSpec(
        node_type=PaymentNodeType.PAYER,
        feature_names=[
            "txn_velocity_1h",        # Count
            "txn_velocity_24h",       # Count
            "avg_amount",             # INR
            "failed_rate",            # Float [0, 1]
            "micro_txn_ratio",        # Float (< INR 100 txns / total)
            "unique_merchants_30d",   # Count
            "account_age_days",       # Float
        ],
        embedding_dim=128,
        description="Payer behavioural profile -- the primary fraud signal carrier",
    ),
    PaymentNodeType.DEVICE: PaymentNodeFeatureSpec(
        node_type=PaymentNodeType.DEVICE,
        feature_names=[
            "unique_payers",          # Count of payers on this device
            "platform",               # Categorical: android/ios/web
            "first_seen_days_ago",    # Float
        ],
        embedding_dim=16,
        description="Device fingerprint node -- shared device = mule network signal",
    ),
    PaymentNodeType.BANK_ACCOUNT: PaymentNodeFeatureSpec(
        node_type=PaymentNodeType.BANK_ACCOUNT,
        feature_names=[
            "unique_merchants",       # Count of merchants sharing this account
            "total_settlement_inr",   # Float
        ],
        embedding_dim=16,
        description="Settlement bank account -- shared account = Ring B signal",
    ),
    PaymentNodeType.IP_ADDRESS: PaymentNodeFeatureSpec(
        node_type=PaymentNodeType.IP_ADDRESS,
        feature_names=[
            "unique_payers",          # Count
            "is_vpn",                 # Boolean
            "country_code",           # Categorical
        ],
        embedding_dim=8,
        description="IP node -- high unique_payers = coordinated attack signal",
    ),
}

PAYMENT_EDGE_WEIGHTS: dict[PaymentEdgeType, PaymentEdgeWeightSpec] = {
    PaymentEdgeType.PAID_TO: PaymentEdgeWeightSpec(
        edge_type=PaymentEdgeType.PAID_TO,
        weight_signals=["txn_count", "total_amount_inr", "refund_count", "last_txn_days_ago"],
        has_temporal_decay=True,
        description="Transaction history between payer and merchant",
    ),
    PaymentEdgeType.SHARES_BANK: PaymentEdgeWeightSpec(
        edge_type=PaymentEdgeType.SHARES_BANK,
        weight_signals=["merchant_count", "total_settlement_inr"],
        has_temporal_decay=False,
        description="Ring B signal: multiple merchants draining to the same bank account",
    ),
    PaymentEdgeType.SHARES_DEVICE: PaymentEdgeWeightSpec(
        edge_type=PaymentEdgeType.SHARES_DEVICE,
        weight_signals=["payer_count", "txn_count_on_device"],
        has_temporal_decay=False,
        description="Ring C signal: multiple payers operating from the same device",
    ),
    PaymentEdgeType.RING_PARTNER: PaymentEdgeWeightSpec(
        edge_type=PaymentEdgeType.RING_PARTNER,
        weight_signals=["shared_payer_count", "shared_bank", "refund_rate_correlation"],
        has_temporal_decay=False,
        description="Derived: two merchants are ring partners if they share >= 3 payers",
    ),
}


def get_pyg_edge_types() -> list[tuple[str, str, str]]:
    """Return edge types as PyG-compatible string tuples."""
    return [(src.value, rel, dst.value) for src, rel, dst in PAYMENT_EDGE_REGISTRY.values()]


def get_dgl_edge_types() -> list[tuple[str, str, str]]:
    """Return edge types as DGL-compatible canonical edge type tuples."""
    return [(src.value, rel, dst.value) for src, rel, dst in PAYMENT_EDGE_REGISTRY.values()]
