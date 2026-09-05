"""
inject_fraud_spikes.py - Controlled synthetic fraud burst injection for reproducible demo spikes.

Three burst types (many small bursts distributed across full date range):
1. inject_card_testing(): Many micro-transactions ($1.00-$5.00) in a tight window.
2. inject_account_takeover(): High-value transactions after sudden device/location switch.
3. inject_velocity_abuse(): Rapid multi-merchant transactions using the same card/instrument.
"""

import os
import sys
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")
CLEAN_CSV = os.path.join(PROCESSED_DIR, "clean_transactions.csv")
INJECTED_CSV = os.path.join(PROCESSED_DIR, "injected_transactions.csv")


def _random_burst_starts(rng: np.random.RandomState, ts_min: pd.Timestamp,
                         ts_max: pd.Timestamp, n_bursts: int) -> list:
    """Generate n_bursts random start timestamps uniformly across [ts_min, ts_max]."""
    total_seconds = int((ts_max - ts_min).total_seconds())
    offsets = rng.randint(0, max(total_seconds, 1), size=n_bursts)
    return sorted([ts_min + pd.Timedelta(seconds=int(s)) for s in offsets])


def inject_card_testing(
    df: pd.DataFrame,
    target_merchant: str = "fraud_Vandervort_Tech",
    n_bursts: int = 25,
    start_time: pd.Timestamp = pd.Timestamp("2026-01-15 14:00:00"),
    seed: int = 42,
    txns_per_burst: int = None,
) -> pd.DataFrame:
    """Inject card testing bursts: rapid micro-transactions over 15 minutes each.

    When called from run_injection_pipeline, n_bursts and start_time are overridden
    to produce many small bursts at random times across the full date range.
    The signature is kept for backward compatibility.
    """
    rng = np.random.RandomState(seed)
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()

    # Determine burst configuration
    if n_bursts >= 20:
        # Pipeline mode: many small bursts at random times
        num_bursts = n_bursts
        txns_per_burst = txns_per_burst or 5
        burst_starts = _random_burst_starts(rng, ts_min, ts_max, num_bursts)
    else:
        # Legacy single-burst mode
        num_bursts = 1
        txns_per_burst = n_bursts
        burst_starts = [start_time]

    burst_rows = []
    for b_idx, burst_start in enumerate(burst_starts):
        card_test_num = f"4532_{seed}_CARDTEST_{b_idx}"
        device_test = f"DEV_TESTING_{seed}_{b_idx}"
        for i in range(txns_per_burst):
            offset_sec = int(i * 30 + rng.randint(0, 10))
            ts = burst_start + pd.Timedelta(seconds=offset_sec)
            amount = round(float(rng.uniform(0.99, 4.99)), 2)
            burst_rows.append({
                "timestamp": ts,
                "merchant": target_merchant,
                "merchant_category": "shopping_net",
                "amount": amount,
                "cardholder_location": "Miami, FL",
                "card_num": card_test_num,
                "device_id": device_test,
                "is_fraud": 1,
                "spike_type": "card_testing",
                "declined": 1 if rng.rand() < 0.7 else 0
            })

    burst_df = pd.DataFrame(burst_rows)
    combined = pd.concat([df, burst_df], ignore_index=True)
    return combined.sort_values("timestamp").reset_index(drop=True)


def inject_account_takeover(
    df: pd.DataFrame,
    target_card: str = None,
    n_bursts: int = 5,
    start_time: pd.Timestamp = pd.Timestamp("2026-01-20 22:15:00"),
    seed: int = 43,
    txns_per_burst: int = None,
) -> pd.DataFrame:
    """Inject account takeover bursts: sudden high-value purchases from foreign device/location.

    When called from run_injection_pipeline, n_bursts and start_time are overridden
    to produce many small bursts at random times across the full date range.
    """
    rng = np.random.RandomState(seed)
    unique_cards = df["card_num"].unique()
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()

    if n_bursts >= 20:
        num_bursts = n_bursts
        txns_per_burst = txns_per_burst or 4
        burst_starts = _random_burst_starts(rng, ts_min, ts_max, num_bursts)
    else:
        num_bursts = 1
        txns_per_burst = n_bursts
        burst_starts = [start_time]

    burst_rows = []
    high_risk_merchants = ["fraud_Kirlin_Inc", "fraud_Vandervort_Tech", "fraud_Cruickshank_Apparel"]

    for b_idx, burst_start in enumerate(burst_starts):
        ato_device = f"DEV_ATO_FOREIGN_{seed}_{b_idx}"
        card_to_use = target_card if target_card is not None else unique_cards[b_idx % len(unique_cards)]
        for i in range(txns_per_burst):
            ts = burst_start + pd.Timedelta(minutes=i * 4 + rng.randint(0, 2))
            merchant = high_risk_merchants[i % len(high_risk_merchants)]
            amount = round(float(rng.uniform(650.0, 2200.0)), 2)
            burst_rows.append({
                "timestamp": ts,
                "merchant": merchant,
                "merchant_category": "shopping_net",
                "amount": amount,
                "cardholder_location": "Foreign/VPN IP",
                "card_num": card_to_use,
                "device_id": ato_device,
                "is_fraud": 1,
                "spike_type": "account_takeover",
                "declined": 0
            })

    burst_df = pd.DataFrame(burst_rows)
    combined = pd.concat([df, burst_df], ignore_index=True)
    return combined.sort_values("timestamp").reset_index(drop=True)


def inject_velocity_abuse(
    df: pd.DataFrame,
    n_bursts: int = 15,
    start_time: pd.Timestamp = pd.Timestamp("2026-01-25 10:00:00"),
    seed: int = 44,
    txns_per_burst: int = None,
) -> pd.DataFrame:
    """Inject velocity abuse bursts: rapid multi-merchant transactions using one payment instrument.

    When called from run_injection_pipeline, n_bursts and start_time are overridden
    to produce many small bursts at random times across the full date range.
    """
    rng = np.random.RandomState(seed)
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()

    if n_bursts >= 20:
        num_bursts = n_bursts
        txns_per_burst = txns_per_burst or 6
        burst_starts = _random_burst_starts(rng, ts_min, ts_max, num_bursts)
    else:
        num_bursts = 1
        txns_per_burst = n_bursts
        burst_starts = [start_time]

    burst_rows = []
    merchants = ["fraud_Rippin_LLC", "fraud_Boyer_Group", "fraud_Kihn_Inc", "fraud_Weber_Market", "fraud_Heller_Gas"]

    for b_idx, burst_start in enumerate(burst_starts):
        velocity_card = f"4532_VELOCITY_ABUSE_{seed}_{b_idx}"
        for i in range(txns_per_burst):
            ts = burst_start + pd.Timedelta(seconds=i * 45 + rng.randint(0, 15))
            merchant = merchants[i % len(merchants)]
            amount = round(float(rng.uniform(120.0, 480.0)), 2)
            burst_rows.append({
                "timestamp": ts,
                "merchant": merchant,
                "merchant_category": "grocery_pos" if i % 2 == 0 else "shopping_net",
                "amount": amount,
                "cardholder_location": "Chicago, IL",
                "card_num": velocity_card,
                "device_id": f"DEV_VEL_{seed}_{b_idx}",
                "is_fraud": 1,
                "spike_type": "velocity_abuse",
                "declined": 0
            })

    burst_df = pd.DataFrame(burst_rows)
    combined = pd.concat([df, burst_df], ignore_index=True)
    return combined.sort_values("timestamp").reset_index(drop=True)


def _compute_burst_counts(base_n: int, target_fraud_rate: float = 0.04):
    """Compute burst counts to keep overall fraud rate in 2-5% range.

    With base_n ~15k transactions (~2.2% organic fraud ~330), we can inject
    up to ~420 more fraud txns to hit ~5%. Using 2 txns per burst, 100 bursts
    per type = 600 injected -> total ~930 fraud in ~15600 txns = ~6% which
    is slightly over, so we scale down to stay under 5%.
    """
    # Estimate organic fraud already in base
    organic_fraud_est = int(base_n * 0.022)

    # Transactions per burst for each type (kept small for more bursts)
    ct_per_burst = 2
    ato_per_burst = 2
    vel_per_burst = 2

    # Start with 100 bursts each
    ct_bursts = 100
    ato_bursts = 100
    vel_bursts = 100

    # Verify and scale to keep fraud rate under 5%
    total_injected = ct_bursts * ct_per_burst + ato_bursts * ato_per_burst + vel_bursts * vel_per_burst
    total_n = base_n + total_injected
    total_fraud = organic_fraud_est + total_injected
    projected_rate = total_fraud / total_n

    if projected_rate > 0.05:
        # Scale down: solve for target 4.8% to strictly stay under 5%
        max_injected = int(0.048 * base_n - organic_fraud_est)
        max_injected = max(max_injected, 60)
        per_type = max_injected // (ct_per_burst + ato_per_burst + vel_per_burst)
        ct_bursts = max(per_type, 20)
        ato_bursts = max(per_type, 20)
        vel_bursts = max(per_type, 20)

    return ct_bursts, ato_bursts, vel_bursts, ct_per_burst, ato_per_burst, vel_per_burst


def _print_train_test_split_summary(df: pd.DataFrame):
    """Print how many fraud examples of each spike_type fall before vs after the 80th percentile timestamp."""
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    n_total = len(df_sorted)
    n_train = int(n_total * 0.8)
    split_ts = df_sorted["timestamp"].iloc[n_train - 1]

    train_df = df_sorted.iloc[:n_train]
    test_df = df_sorted.iloc[n_train:]

    print(f"\n{'='*70}")
    print(f"  TRAIN/TEST SPLIT FRAUD DISTRIBUTION SUMMARY")
    print(f"  80th percentile timestamp: {split_ts}")
    print(f"  Train set: {len(train_df)} txns | Test set: {len(test_df)} txns")
    print(f"{'='*70}")

    spike_types = ["organic", "card_testing", "account_takeover", "velocity_abuse"]
    header = f"  {'spike_type':<22} {'Train Fraud':>12} {'Test Fraud':>12} {'Total':>8} {'Test %':>8}"
    print(header)
    print(f"  {'-'*62}")

    for st in spike_types:
        train_count = int(((train_df["spike_type"] == st) & (train_df["is_fraud"] == 1)).sum())
        test_count = int(((test_df["spike_type"] == st) & (test_df["is_fraud"] == 1)).sum())
        total = train_count + test_count
        test_pct = f"{test_count / total * 100:.1f}%" if total > 0 else "N/A"
        print(f"  {st:<22} {train_count:>12} {test_count:>12} {total:>8} {test_pct:>8}")

    # Totals
    train_fraud = int(train_df["is_fraud"].sum())
    test_fraud = int(test_df["is_fraud"].sum())
    total_fraud = train_fraud + test_fraud
    test_pct_total = f"{test_fraud / total_fraud * 100:.1f}%" if total_fraud > 0 else "N/A"
    print(f"  {'-'*62}")
    print(f"  {'TOTAL':<22} {train_fraud:>12} {test_fraud:>12} {total_fraud:>8} {test_pct_total:>8}")
    print(f"{'='*70}\n")


def run_injection_pipeline(clean_csv_path: str = CLEAN_CSV, seed: int = 42) -> pd.DataFrame:
    """Execute all fraud burst injections deterministically."""
    print(f"[inject_fraud_spikes] Loading clean base data from {clean_csv_path}...")
    if not os.path.exists(clean_csv_path):
        from src.person_a.clean_data import main as clean_main
        clean_main()

    df = pd.read_csv(clean_csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "spike_type" not in df.columns:
        df["spike_type"] = "organic"
    if "declined" not in df.columns:
        df["declined"] = 0

    # Compute burst counts to keep fraud rate in 2-5% range
    ct_bursts, ato_bursts, vel_bursts, ct_tpb, ato_tpb, vel_tpb = _compute_burst_counts(len(df))
    print(f"[inject_fraud_spikes] Burst counts: card_testing={ct_bursts} ({ct_tpb}/burst), "
          f"account_takeover={ato_bursts} ({ato_tpb}/burst), velocity_abuse={vel_bursts} ({vel_tpb}/burst)")

    print("[inject_fraud_spikes] Injecting Card Testing bursts...")
    df = inject_card_testing(df, n_bursts=ct_bursts, seed=seed, txns_per_burst=ct_tpb)

    print("[inject_fraud_spikes] Injecting Account Takeover bursts...")
    df = inject_account_takeover(df, n_bursts=ato_bursts, seed=seed + 1, txns_per_burst=ato_tpb)

    print("[inject_fraud_spikes] Injecting Velocity Abuse bursts...")
    df = inject_velocity_abuse(df, n_bursts=vel_bursts, seed=seed + 2, txns_per_burst=vel_tpb)

    # Re-assign clean IDs
    df["transaction_id"] = [f"TXN-{i+10001:06d}" for i in range(len(df))]
    print(f"[inject_fraud_spikes] Final dataset shape after injection: {df.shape}")
    print(f"[inject_fraud_spikes] Total fraud count: {df['is_fraud'].sum()} ({df['is_fraud'].mean():.2%})")

    # Print train/test split summary to verify coverage
    _print_train_test_split_summary(df)

    return df


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df = run_injection_pipeline()
    df.to_csv(INJECTED_CSV, index=False)
    print(f"[inject_fraud_spikes] [OK] Saved injected transactions to {INJECTED_CSV}")


if __name__ == "__main__":
    main()
