"""Razorpay payment graph data generator.

Mirrors src/data/generate_graph.py but for the payments domain.
Generates three fraud rings in a realistic Razorpay-shaped dataset:
  Ring A: Refund cycling (3 merchants, 15 payers)
  Ring B: Shared bank account (5 merchants, same settlement IBAN)
  Ring C: Card testing velocity burst (1 mule payer, 35 micro-txns)

Output:
  razorpay_merchants.csv
  razorpay_payers.csv
  interactions/razorpay_transactions.csv
"""

import hashlib
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from faker import Faker

fake = Faker("en_IN")
random.seed(42)

NUM_MERCHANTS = 80
NUM_PAYERS = 400
NUM_CLEAN = 2000
BASE_TS = datetime(2024, 10, 1)


def _ts(offset_hours: float) -> str:
    return (BASE_TS + timedelta(hours=offset_hours)).isoformat()


def _device(seed: int) -> str:
    return hashlib.md5(f"device_{seed}".encode()).hexdigest()[:16]


# ── Entity pools ──────────────────────────────────────────────────────────────


def _make_merchants() -> pd.DataFrame:
    rows = []
    for i in range(NUM_MERCHANTS):
        bank = f"BANKA_{i:04d}"
        ring = None
        if 5 <= i < 10:
            bank = "SHARED_BANK_XYZ"
            ring = "RING_B"
        elif 0 <= i < 3:
            ring = "RING_A"
        rows.append(
            {
                "merchant_id": f"mrc_{i:05d}",
                "name": fake.company(),
                "category": random.choice(["retail", "food", "travel", "edtech", "saas", "gaming"]),
                "bank_account": bank,
                "fraud_ring": ring,
                "created_at": _ts(-random.uniform(200, 5000)),
            }
        )
    return pd.DataFrame(rows)


def _make_payers() -> pd.DataFrame:
    rows = []
    for i in range(NUM_PAYERS):
        ring = None
        if i < 15:
            ring = "RING_A"
        elif i == 200:
            ring = "RING_C"
        rows.append(
            {
                "payer_id": f"pay_{i:05d}",
                "email": fake.email(),
                "phone": f"9{random.randint(100000000, 999999999)}",
                "device_id": _device(i // 2),  # every 2 payers share a device
                "ip": fake.ipv4(),
                "fraud_ring": ring,
            }
        )
    return pd.DataFrame(rows)


def _make_clean_txns(merchants: pd.DataFrame, payers: pd.DataFrame) -> list[dict]:
    mids = merchants["merchant_id"].tolist()
    pids = payers["payer_id"].tolist()
    rows = []
    for i in range(NUM_CLEAN):
        rows.append(
            {
                "payment_id": f"pay_txn_{i:05d}",
                "merchant_id": random.choice(mids),
                "payer_id": random.choice(pids),
                "amount": round(random.uniform(199, 12000), 2),
                "status": random.choices(["captured", "refunded", "failed"], weights=[85, 10, 5])[0],
                "method": random.choice(["upi", "card", "netbanking", "wallet"]),
                "device_id": _device(random.randint(0, 200)),
                "created_at": _ts(random.uniform(0, 720)),
                "fraud_label": 0,
            }
        )
    return rows


def _inject_ring_a(merchants: pd.DataFrame, payers: pd.DataFrame) -> list[dict]:
    ring_m = merchants[merchants["fraud_ring"] == "RING_A"]["merchant_id"].tolist()
    ring_p = payers[payers["fraud_ring"] == "RING_A"]["payer_id"].tolist()
    rows = []
    for cycle in range(40):
        m = random.choice(ring_m)
        p = random.choice(ring_p)
        t = random.uniform(0, 400)
        amt = round(random.uniform(8000, 50000), 2)
        base_i = 90000 + cycle * 2
        rows.append(
            {
                "payment_id": f"pay_txn_{base_i}",
                "merchant_id": m,
                "payer_id": p,
                "amount": amt,
                "status": "captured",
                "method": "upi",
                "device_id": _device(random.randint(0, 14)),
                "created_at": _ts(t),
                "fraud_label": 1,
            }
        )
        rows.append(
            {
                "payment_id": f"pay_txn_{base_i + 1}",
                "merchant_id": m,
                "payer_id": p,
                "amount": round(amt * 0.97, 2),
                "status": "refunded",
                "method": "upi",
                "device_id": _device(random.randint(0, 14)),
                "created_at": _ts(t + random.uniform(0.5, 3)),
                "fraud_label": 1,
            }
        )
    return rows


def _inject_ring_b(merchants: pd.DataFrame, payers: pd.DataFrame) -> list[dict]:
    ring_m = merchants[merchants["fraud_ring"] == "RING_B"]["merchant_id"].tolist()
    rows = []
    for i in range(60):
        m = random.choice(ring_m)
        p = payers.iloc[20 + (i % 60)]["payer_id"]
        rows.append(
            {
                "payment_id": f"pay_txn_9{1000 + i}",
                "merchant_id": m,
                "payer_id": p,
                "amount": round(random.uniform(500, 5000), 2),
                "status": "captured",
                "method": random.choice(["card", "upi"]),
                "device_id": _device(i % 20),
                "created_at": _ts(random.uniform(10, 600)),
                "fraud_label": 1,
            }
        )
    return rows


def _inject_ring_c(payers: pd.DataFrame, merchants: pd.DataFrame) -> list[dict]:
    mule = payers[payers["fraud_ring"] == "RING_C"].iloc[0]["payer_id"]
    dev = _device(999)
    mids = merchants.iloc[10:30]["merchant_id"].tolist()
    rows = []
    t_base = 300.0
    for i in range(35):
        rows.append(
            {
                "payment_id": f"pay_txn_9{2000 + i}",
                "merchant_id": random.choice(mids),
                "payer_id": mule,
                "amount": round(random.uniform(1, 50), 2),
                "status": random.choices(["captured", "failed"], weights=[60, 40])[0],
                "method": "card",
                "device_id": dev,
                "created_at": _ts(t_base + i * 0.01),
                "fraud_label": 1,
            }
        )
    return rows


def generate(output_dir: str | Path = ".") -> None:
    out = Path(output_dir)
    (out / "interactions").mkdir(parents=True, exist_ok=True)

    merchants = _make_merchants()
    payers = _make_payers()

    all_txns = (
        _make_clean_txns(merchants, payers)
        + _inject_ring_a(merchants, payers)
        + _inject_ring_b(merchants, payers)
        + _inject_ring_c(payers, merchants)
    )
    random.shuffle(all_txns)
    txns_df = pd.DataFrame(all_txns)

    merchants.to_csv(out / "razorpay_merchants.csv", index=False)
    payers.to_csv(out / "razorpay_payers.csv", index=False)
    txns_df.to_csv(out / "interactions" / "razorpay_transactions.csv", index=False)

    fraud = txns_df["fraud_label"].sum()
    print(f"Generated {len(txns_df)} transactions | {fraud} fraud ({fraud / len(txns_df) * 100:.1f}%)")
    print(f"  Ring A: {len([t for t in all_txns if t.get('fraud_label') and 'ring_a' not in str(t).lower()])} txns")
    print(f"Saved to {out}/")


if __name__ == "__main__":
    generate(Path(__file__).parent)
