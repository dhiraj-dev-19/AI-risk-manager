"""
decision_engine.py - Defense-Only Decision & Risk-Band Mapping Engine.

Maps ML risk score -> risk band -> strictly defense-only action:
- Score < threshold: allow
- Medium band: flag_for_review
- High band: hold_for_verification
- Very high band: auto_decline

Strict Guardrail: Action enum is defense-only. No external party contact or automated retaliation.
Consumes thresholds dynamically from threshold_analysis.json.
"""

import os
import sys
import json
import time
from typing import Dict, Any

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

THRESHOLD_JSON = os.path.join(_PROJECT_ROOT, "data", "models", "threshold_analysis.json")

# Defensive Action Enum (Strict Hackathon Constraint)
DEFENSE_ACTIONS = ["allow", "flag_for_review", "hold_for_verification", "auto_decline"]


def load_threshold_config() -> Dict[str, Any]:
    """Load threshold configuration from threshold_analysis.json or use defaults."""
    if os.path.exists(THRESHOLD_JSON):
        try:
            with open(THRESHOLD_JSON, "r") as f:
                data = json.load(f)
                return data
        except Exception as e:
            print(f"[decision_engine] Warning reading threshold_analysis.json: {e}")

    # Default fallback bands if file not present
    return {
        "optimal_threshold": 0.35,
        "recommended_bands": {
            "low": {"max": 0.35, "action": "allow"},
            "medium": {"min": 0.35, "max": 0.60, "action": "flag_for_review"},
            "high": {"min": 0.60, "max": 0.85, "action": "hold_for_verification"},
            "very_high": {"min": 0.85, "max": 1.00, "action": "auto_decline"}
        }
    }


def evaluate_decision(scored_transaction: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate decision engine logic on a scored transaction dictionary."""
    score = float(scored_transaction.get("risk_score", 0.0))
    tx_id = scored_transaction.get("transaction_id", f"TXN-{int(time.time()*1000):06d}")
    ts = str(scored_transaction.get("timestamp", pd.Timestamp.now() if 'pd' in globals() else "2026-01-01"))

    cfg = load_threshold_config()
    threshold = float(cfg.get("optimal_threshold", 0.35))

    # Map score to band and action
    if score >= 0.85:
        risk_band = "very_high"
        action = "auto_decline"
    elif score >= max(threshold, 0.70):
        risk_band = "high"
        action = "hold_for_verification"
    elif score >= threshold:
        risk_band = "medium"
        action = "flag_for_review"
    else:
        risk_band = "low"
        action = "allow"

    # Enforce strict defense-only action constraint
    if action not in DEFENSE_ACTIONS:
        action = "flag_for_review"

    # Extract top contributing features (deltas/importance)
    features = scored_transaction.get("features", {})
    top_features = {}
    if features:
        # Sort features by absolute deviation/value for explainability signal
        sorted_feats = sorted(
            features.items(),
            key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
            reverse=True
        )
        top_features = dict(sorted_feats[:4])

    amount = float(scored_transaction.get("amount", 0.0))
    est_fp_cost = round(amount * 1.1, 2) if action != "allow" else 0.0
    est_fraud_caught = round(amount, 2) if (action != "allow" and score >= threshold) else 0.0

    return {
        "transaction_id": tx_id,
        "timestamp": ts,
        "merchant": scored_transaction.get("merchant", "unknown"),
        "merchant_category": scored_transaction.get("merchant_category", "grocery_pos"),
        "amount": amount,
        "card_num": scored_transaction.get("card_num", ""),
        "device_id": scored_transaction.get("device_id", ""),
        "risk_score": score,
        "risk_band": risk_band,
        "action": action,
        "cohort_context": scored_transaction.get("cohort_context", {}),
        "top_features": top_features,
        "threshold_used": threshold,
        "estimated_fp_cost_at_threshold": est_fp_cost,
        "estimated_fraud_caught_at_threshold": est_fraud_caught
    }


if __name__ == "__main__":
    dummy_scored = {
        "transaction_id": "TXN-TEST-001",
        "timestamp": "2026-01-20 16:00:00",
        "merchant": "fraud_Vandervort_Tech",
        "merchant_category": "shopping_net",
        "amount": 1250.00,
        "risk_score": 0.78,
        "features": {"amount_baseline_zscore": 3.8, "distinct_cards_per_device_24h": 4.0},
        "cohort_context": {"historical_mean_amount": 65.0, "amount_ratio_vs_baseline": 19.23}
    }
    decision = evaluate_decision(dummy_scored)
    print("Decision Output:")
    print(json.dumps(decision, indent=2))
