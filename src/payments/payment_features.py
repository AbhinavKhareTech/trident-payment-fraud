"""Feature engineering for Prong 3 (XGBoost) on the payment fraud graph.

Mirrors the consumption graph XGBoostFeatureExtractor pattern but for
fraud detection: velocity windows, amount anomaly, refund cycling,
shared-entity flags derived from the payment graph adjacency.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd


class PaymentFraudFeatureExtractor:
    """Extract tabular fraud features from raw transaction data.

    Designed to be called from PaymentGraphBuilder.get_xgboost_features()
    and independently as a streaming feature pipeline for real-time scoring.
    """

    def __init__(
        self,
        transactions: pd.DataFrame,
        merchants: pd.DataFrame,
        payers: pd.DataFrame,
    ) -> None:
        self.txns = transactions
        self.merchants = merchants
        self.payers = payers

        self._now = datetime.utcnow()
        # Pre-index for fast per-entity lookup
        self._payer_txns = transactions.groupby("payer_id")
        self._merchant_txns = transactions.groupby("merchant_id")

    # ── Payer velocity features ───────────────────────────────────────────────

    def extract_payer_features(self) -> pd.DataFrame:
        """One row per payer: velocity windows, amount stats, failed/micro rates."""
        rows = []
        for payer_id, group in self._payer_txns:
            group = group.copy()
            group["created_at"] = pd.to_datetime(group["created_at"])
            latest = group["created_at"].max()

            # Velocity windows (count in last N hours)
            def velocity(hours: int) -> int:
                cutoff = latest - pd.Timedelta(hours=hours)
                return int((group["created_at"] >= cutoff).sum())

            amounts = group["amount"].values
            failed = (group["status"] == "failed").sum()
            refunded = (group["status"] == "refunded").sum()
            micro = (group["amount"] < 100).sum()
            n = len(group)

            # Amount z-score against self-history
            zscore = 0.0
            if len(amounts) >= 3:
                std = amounts.std() or 1.0
                zscore = abs(amounts[-1] - amounts.mean()) / std

            rows.append(
                {
                    "payer_id": payer_id,
                    "txn_count": n,
                    "velocity_1h": velocity(1),
                    "velocity_6h": velocity(6),
                    "velocity_24h": velocity(24),
                    "avg_amount": float(np.mean(amounts)),
                    "max_amount": float(np.max(amounts)),
                    "amount_zscore": round(zscore, 3),
                    "failed_rate": round(failed / n, 3),
                    "refund_rate": round(refunded / n, 3),
                    "micro_txn_ratio": round(micro / n, 3),
                    "unique_merchants": group["merchant_id"].nunique(),
                    "unique_devices": group["device_id"].nunique() if "device_id" in group else 0,
                }
            )
        return pd.DataFrame(rows)

    # ── Merchant features ─────────────────────────────────────────────────────

    def extract_merchant_features(self) -> pd.DataFrame:
        """One row per merchant: refund rate, velocity, settlement signals."""
        rows = []
        for merchant_id, group in self._merchant_txns:
            n = len(group)
            refunded = (group["status"] == "refunded").sum()
            captured = (group["status"] == "captured").sum()
            refund_val = group.loc[group["status"] == "refunded", "amount"].sum()
            captured_val = group.loc[group["status"] == "captured", "amount"].sum()

            rows.append(
                {
                    "merchant_id": merchant_id,
                    "txn_count": n,
                    "refund_rate": round(refunded / n, 3),
                    "refund_value_ratio": round(refund_val / captured_val, 3) if captured_val else 0,
                    "avg_amount": round(group["amount"].mean(), 2),
                    "unique_payers": group["payer_id"].nunique(),
                    "captured_volume_inr": round(float(captured_val), 2),
                }
            )

        # Merge bank-sharing count from merchants table
        bank_counts = self.merchants.groupby("bank_account")["merchant_id"].transform("count").rename("merchants_sharing_bank")
        merchant_meta = self.merchants.copy()
        merchant_meta["merchants_sharing_bank"] = bank_counts

        result = pd.DataFrame(rows).merge(
            merchant_meta[["merchant_id", "merchants_sharing_bank", "category"]],
            on="merchant_id",
            how="left",
        )
        result["shared_bank_flag"] = (result["merchants_sharing_bank"] > 1).astype(int)
        return result

    # ── Pair-level features for ensemble ─────────────────────────────────────

    def extract_pair_features(self) -> pd.DataFrame:
        """One row per (payer, merchant) pair: combined risk signals."""
        payer_feats = self.extract_payer_features().set_index("payer_id")
        merchant_feats = self.extract_merchant_features().set_index("merchant_id")

        pair_agg = (
            self.txns.groupby(["payer_id", "merchant_id"])
            .agg(
                pair_txn_count=("payment_id", "size"),
                pair_avg_amount=("amount", "mean"),
                pair_refunds=("status", lambda x: (x == "refunded").sum()),
            )
            .reset_index()
        )

        pair_agg = pair_agg.merge(payer_feats.add_prefix("payer_"), left_on="payer_id", right_index=True, how="left").merge(
            merchant_feats.add_prefix("merchant_"), left_on="merchant_id", right_index=True, how="left"
        )

        # Fraud label from transactions (if available)
        labels = (
            self.txns.groupby(["payer_id", "merchant_id"])["fraud_label"].max().reset_index()
            if "fraud_label" in self.txns.columns
            else None
        )
        if labels is not None:
            pair_agg = pair_agg.merge(labels, on=["payer_id", "merchant_id"], how="left")
            pair_agg["fraud_label"] = pair_agg["fraud_label"].fillna(0).astype(int)

        return pair_agg
