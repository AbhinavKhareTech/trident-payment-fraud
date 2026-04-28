"""Heterogeneous payment fraud graph construction.

Mirrors ConsumptionGraphBuilder (builder.py) but for the Razorpay
payments domain.  Builds a single payment graph with 5 node types
and 7 edge types, then exports to PyG, DGL, and XGBoost formats
for Prong 1, Prong 2, and Prong 3 respectively.

The cross-entity edges (SHARES_BANK, RING_PARTNER, CO_PAYER) are
derived during build -- they are not in the raw transaction data.
This is the key capability that single-record fraud engines lack.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import torch
    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    _TORCH_AVAILABLE = False
    # Fallback: represent tensors as numpy arrays for non-GNN paths
    class _TorchStub:
        @staticmethod
        def tensor(data, dtype=None):
            return np.array(data)
        @staticmethod
        def zeros(*shape):
            return np.zeros(shape)
        long   = None
        float32= None
    torch = _TorchStub()  # type: ignore

from bgi_trident.graph.schema_payments import (
    PAYMENT_EDGE_REGISTRY,
    PAYMENT_NODE_FEATURES,
    PaymentEdgeType,
    PaymentNodeType,
)

logger = logging.getLogger(__name__)


@dataclass
class PaymentInteractionData:
    """Raw payment data loaded from CSVs."""

    transactions: pd.DataFrame  # payment_id, merchant_id, payer_id, amount, status, method, created_at, device_id, ip, fraud_label
    merchants:    pd.DataFrame  # merchant_id, name, category, bank_account, created_at
    payers:       pd.DataFrame  # payer_id, email, phone, device_id, ip


class PaymentGraphBuilder:
    """Builds the heterogeneous payment fraud graph.

    Same build pipeline as ConsumptionGraphBuilder:
    nodes -> direct edges -> derived cross-entity edges -> tensor export.
    """

    def __init__(self, data: PaymentInteractionData) -> None:
        self.data = data
        self._node_maps: dict[PaymentNodeType, dict[str, int]] = {}
        self._node_features: dict[str, torch.Tensor] = {}
        self._edge_indices: dict[tuple[str, str, str], torch.Tensor] = {}
        self._edge_weights: dict[tuple[str, str, str], torch.Tensor] = {}

        # NetworkX-style adjacency for fraud logic (not for tensor export)
        self._merchant_txns:   dict[str, list[dict]] = defaultdict(list)
        self._payer_txns:      dict[str, list[dict]] = defaultdict(list)
        self._bank_merchants:  dict[str, set[str]]   = defaultdict(set)
        self._device_payers:   dict[str, set[str]]   = defaultdict(set)

    def build(self) -> PaymentGraphBuilder:
        """Full build pipeline."""
        logger.info("Building payment fraud graph...")
        self._build_node_maps()
        self._index_transactions()
        self._build_node_features()
        self._build_paid_to_edges()
        self._build_bank_edges()
        self._build_device_ip_edges()
        self._derive_ring_partner_edges()
        self._derive_co_payer_edges()
        logger.info(
            "Payment graph built: %d node types, %d edge types",
            len(self._node_features),
            len(self._edge_indices),
        )
        return self

    # ── Node maps ─────────────────────────────────────────────────────────────

    def _build_node_maps(self) -> None:
        self._node_maps[PaymentNodeType.MERCHANT] = {
            mid: idx for idx, mid in enumerate(self.data.merchants["merchant_id"].unique())
        }
        self._node_maps[PaymentNodeType.PAYER] = {
            pid: idx for idx, pid in enumerate(self.data.payers["payer_id"].unique())
        }
        devices = self.data.payers["device_id"].dropna().unique()
        self._node_maps[PaymentNodeType.DEVICE] = {d: i for i, d in enumerate(devices)}

        banks = self.data.merchants["bank_account"].dropna().unique()
        self._node_maps[PaymentNodeType.BANK_ACCOUNT] = {b: i for i, b in enumerate(banks)}

        ips = self.data.payers["ip"].dropna().unique()
        self._node_maps[PaymentNodeType.IP_ADDRESS] = {ip: i for i, ip in enumerate(ips)}

    def _index_transactions(self) -> None:
        """Index transactions by merchant and payer for fast lookup."""
        for _, row in self.data.transactions.iterrows():
            t = row.to_dict()
            self._merchant_txns[t["merchant_id"]].append(t)
            self._payer_txns[t["payer_id"]].append(t)

        # Index bank account -> merchant set
        for _, row in self.data.merchants.iterrows():
            self._bank_merchants[row["bank_account"]].add(row["merchant_id"])

        # Index device -> payer set
        for _, row in self.data.payers.iterrows():
            if pd.notna(row.get("device_id")):
                self._device_payers[row["device_id"]].add(row["payer_id"])

    # ── Node features ─────────────────────────────────────────────────────────

    def _build_node_features(self) -> None:
        self._node_features["merchant"]     = self._merchant_features()
        self._node_features["payer"]        = self._payer_features()
        self._node_features["device"]       = self._device_features()
        self._node_features["bank_account"] = self._bank_features()
        self._node_features["ip_address"]   = self._ip_features()

    def _merchant_features(self) -> torch.Tensor:
        n   = len(self._node_maps[PaymentNodeType.MERCHANT])
        dim = len(PAYMENT_NODE_FEATURES[PaymentNodeType.MERCHANT].feature_names)
        feats = np.zeros((n, dim), dtype=np.float32)
        for _, row in self.data.merchants.iterrows():
            idx = self._node_maps[PaymentNodeType.MERCHANT].get(row["merchant_id"], -1)
            if idx < 0:
                continue
            txns = self._merchant_txns[row["merchant_id"]]
            if txns:
                refunded = sum(1 for t in txns if t.get("status") == "refunded")
                feats[idx, 1] = refunded / len(txns)           # refund_rate
                feats[idx, 2] = np.mean([t["amount"] for t in txns])   # avg_txn_amount
                feats[idx, 3] = len(txns)                        # txn_velocity proxy
        return torch.tensor(feats)

    def _payer_features(self) -> torch.Tensor:
        n   = len(self._node_maps[PaymentNodeType.PAYER])
        dim = len(PAYMENT_NODE_FEATURES[PaymentNodeType.PAYER].feature_names)
        feats = np.zeros((n, dim), dtype=np.float32)
        for _, row in self.data.payers.iterrows():
            idx = self._node_maps[PaymentNodeType.PAYER].get(row["payer_id"], -1)
            if idx < 0:
                continue
            txns = self._payer_txns[row["payer_id"]]
            if txns:
                failed = sum(1 for t in txns if t.get("status") == "failed")
                micro  = sum(1 for t in txns if t.get("amount", 0) < 100)
                feats[idx, 1] = len(txns)                              # velocity proxy
                feats[idx, 3] = np.mean([t["amount"] for t in txns])  # avg_amount
                feats[idx, 4] = failed / len(txns)                     # failed_rate
                feats[idx, 5] = micro  / len(txns)                     # micro_txn_ratio
        return torch.tensor(feats)

    def _device_features(self) -> torch.Tensor:
        n   = len(self._node_maps[PaymentNodeType.DEVICE])
        dim = len(PAYMENT_NODE_FEATURES[PaymentNodeType.DEVICE].feature_names)
        feats = np.zeros((n, dim), dtype=np.float32)
        for dev_id, payer_set in self._device_payers.items():
            idx = self._node_maps[PaymentNodeType.DEVICE].get(dev_id, -1)
            if idx >= 0:
                feats[idx, 0] = len(payer_set)
        return torch.tensor(feats)

    def _bank_features(self) -> torch.Tensor:
        n   = len(self._node_maps[PaymentNodeType.BANK_ACCOUNT])
        dim = len(PAYMENT_NODE_FEATURES[PaymentNodeType.BANK_ACCOUNT].feature_names)
        feats = np.zeros((n, dim), dtype=np.float32)
        for bank_id, merchant_set in self._bank_merchants.items():
            idx = self._node_maps[PaymentNodeType.BANK_ACCOUNT].get(bank_id, -1)
            if idx >= 0:
                feats[idx, 0] = len(merchant_set)
        return torch.tensor(feats)

    def _ip_features(self) -> torch.Tensor:
        n   = len(self._node_maps[PaymentNodeType.IP_ADDRESS])
        dim = len(PAYMENT_NODE_FEATURES[PaymentNodeType.IP_ADDRESS].feature_names)
        return torch.zeros(n, dim)

    # ── Edges ─────────────────────────────────────────────────────────────────

    def _build_paid_to_edges(self) -> None:
        """Payer -> Merchant: PAID_TO (aggregated per payer-merchant pair)."""
        agg = (
            self.data.transactions
            .groupby(["payer_id", "merchant_id"])
            .agg(
                txn_count=("payment_id", "size"),
                total_amount=("amount", "sum"),
                refund_count=("status", lambda x: (x == "refunded").sum()),
            )
            .reset_index()
        )
        src, dst, weights = [], [], []
        for _, row in agg.iterrows():
            s = self._node_maps[PaymentNodeType.PAYER].get(row["payer_id"],    -1)
            d = self._node_maps[PaymentNodeType.MERCHANT].get(row["merchant_id"], -1)
            if s >= 0 and d >= 0:
                src.append(s); dst.append(d)
                weights.append(row["txn_count"])
        if src:
            key = ("payer", "paid_to", "merchant")
            self._edge_indices[key] = torch.tensor([src, dst], dtype=torch.long)
            self._edge_weights[key]  = torch.tensor(weights,   dtype=torch.float32)

    def _build_bank_edges(self) -> None:
        """Merchant -> BankAccount: SHARES_BANK."""
        src, dst = [], []
        for _, row in self.data.merchants.iterrows():
            s = self._node_maps[PaymentNodeType.MERCHANT].get(row["merchant_id"], -1)
            d = self._node_maps[PaymentNodeType.BANK_ACCOUNT].get(row["bank_account"], -1)
            if s >= 0 and d >= 0:
                src.append(s); dst.append(d)
        if src:
            key = ("merchant", "shares_bank", "bank_account")
            self._edge_indices[key] = torch.tensor([src, dst], dtype=torch.long)

    def _build_device_ip_edges(self) -> None:
        """Payer -> Device: SHARES_DEVICE. Payer -> IP: SAME_IP."""
        dev_src, dev_dst = [], []
        ip_src,  ip_dst  = [], []
        for _, row in self.data.payers.iterrows():
            s = self._node_maps[PaymentNodeType.PAYER].get(row["payer_id"], -1)
            if s < 0:
                continue
            if pd.notna(row.get("device_id")):
                d = self._node_maps[PaymentNodeType.DEVICE].get(row["device_id"], -1)
                if d >= 0:
                    dev_src.append(s); dev_dst.append(d)
            if pd.notna(row.get("ip")):
                i = self._node_maps[PaymentNodeType.IP_ADDRESS].get(row["ip"], -1)
                if i >= 0:
                    ip_src.append(s); ip_dst.append(i)
        if dev_src:
            self._edge_indices[("payer", "shares_device", "device")] = torch.tensor([dev_src, dev_dst], dtype=torch.long)
        if ip_src:
            self._edge_indices[("payer", "same_ip", "ip_address")] = torch.tensor([ip_src, ip_dst], dtype=torch.long)

    def _derive_ring_partner_edges(self, min_shared: int = 3) -> None:
        """Merchant -> Merchant: RING_PARTNER (derived via bipartite projection).

        Two merchants are ring partners if >= min_shared payers paid both.
        This is the FOLLOWED_BY_DINING equivalent for fraud graphs.
        """
        # Build payer -> {merchants} index
        payer_to_merchants: dict[str, set[str]] = defaultdict(set)
        for _, row in self.data.transactions.iterrows():
            payer_to_merchants[row["payer_id"]].add(row["merchant_id"])

        # Co-occurrence count
        pair_count: dict[tuple[str, str], int] = defaultdict(int)
        for merchants in payer_to_merchants.values():
            ml = sorted(merchants)
            for i in range(len(ml)):
                for j in range(i + 1, len(ml)):
                    pair_count[(ml[i], ml[j])] += 1

        src, dst, weights = [], [], []
        for (m1, m2), cnt in pair_count.items():
            if cnt < min_shared:
                continue
            s = self._node_maps[PaymentNodeType.MERCHANT].get(m1, -1)
            d = self._node_maps[PaymentNodeType.MERCHANT].get(m2, -1)
            if s >= 0 and d >= 0:
                src.extend([s, d]); dst.extend([d, s])  # undirected
                weights.extend([cnt, cnt])

        if src:
            key = ("merchant", "ring_partner", "merchant")
            self._edge_indices[key] = torch.tensor([src, dst], dtype=torch.long)
            self._edge_weights[key]  = torch.tensor(weights, dtype=torch.float32)

    def _derive_co_payer_edges(self) -> None:
        """Payer -> Payer: CO_PAYER (same device, different account).

        Signals a mule network operating from a single device.
        """
        src, dst = [], []
        for payer_set in self._device_payers.values():
            payers = sorted(payer_set)
            for i in range(len(payers)):
                for j in range(i + 1, len(payers)):
                    s = self._node_maps[PaymentNodeType.PAYER].get(payers[i], -1)
                    d = self._node_maps[PaymentNodeType.PAYER].get(payers[j], -1)
                    if s >= 0 and d >= 0:
                        src.extend([s, d]); dst.extend([d, s])
        if src:
            key = ("payer", "co_payer", "payer")
            self._edge_indices[key] = torch.tensor([src, dst], dtype=torch.long)

    # ── Exports ───────────────────────────────────────────────────────────────

    def to_pyg(self):
        """Export to PyG HeteroData for Prong 1 (structural GNN)."""
        from torch_geometric.data import HeteroData
        data = HeteroData()
        for nt, feats in self._node_features.items():
            data[nt].x = feats
            data[nt].num_nodes = feats.size(0)
        for edge_key, edge_index in self._edge_indices.items():
            data[edge_key].edge_index = edge_index
            if edge_key in self._edge_weights:
                data[edge_key].edge_weight = self._edge_weights[edge_key]
        return data

    def to_dgl(self):
        """Export to DGL DGLHeteroGraph for Prong 2 (temporal R-GCN)."""
        import dgl
        graph_data = {ek: (ei[0], ei[1]) for ek, ei in self._edge_indices.items()}
        if not graph_data:
            raise ValueError("No edges. Cannot build DGL graph.")
        g = dgl.heterograph(graph_data)
        for nt, feats in self._node_features.items():
            if nt in g.ntypes:
                g.nodes[nt].data["feat"] = feats
        for ek, w in self._edge_weights.items():
            if ek in g.canonical_etypes:
                g.edges[ek].data["weight"] = w
        return g

    def get_xgboost_features(self) -> pd.DataFrame:
        """Export tabular fraud features for Prong 3.

        One row per (payer, merchant) pair with velocity, amount, refund signals.
        """
        features = []
        agg = (
            self.data.transactions
            .groupby(["payer_id", "merchant_id"])
            .agg(
                txn_count       = ("payment_id", "size"),
                total_amount    = ("amount",     "sum"),
                avg_amount      = ("amount",     "mean"),
                refund_count    = ("status",     lambda x: (x == "refunded").sum()),
                failed_count    = ("status",     lambda x: (x == "failed").sum()),
                micro_count     = ("amount",     lambda x: (x < 100).sum()),
                last_txn        = ("created_at", "max"),
            )
            .reset_index()
        )
        for _, row in agg.iterrows():
            features.append({
                "payer_id":       row["payer_id"],
                "merchant_id":    row["merchant_id"],
                "txn_count":      row["txn_count"],
                "avg_amount":     row["avg_amount"],
                "refund_rate":    row["refund_count"] / row["txn_count"],
                "failed_rate":    row["failed_count"] / row["txn_count"],
                "micro_txn_ratio":row["micro_count"]  / row["txn_count"],
                "total_amount":   row["total_amount"],
            })
        return pd.DataFrame(features)

    @property
    def node_counts(self) -> dict[str, int]:
        return {nt.value: len(nm) for nt, nm in self._node_maps.items()}

    @property
    def edge_counts(self) -> dict[str, int]:
        result = {}
        for (s, r, d), idx in self._edge_indices.items():
            key = f"{s}-{r}->{d}"
            try:
                result[key] = idx.shape[1]    # torch tensor
            except (AttributeError, IndexError):
                result[key] = int(np.array(idx).shape[-1]) if hasattr(idx, 'shape') else 0
        return result

    # Public accessors for fraud tools
    @property
    def merchant_txns(self) -> dict[str, list[dict]]:
        return dict(self._merchant_txns)

    @property
    def payer_txns(self) -> dict[str, list[dict]]:
        return dict(self._payer_txns)

    @property
    def bank_merchants(self) -> dict[str, set[str]]:
        return dict(self._bank_merchants)

    @property
    def device_payers(self) -> dict[str, set[str]]:
        return dict(self._device_payers)
