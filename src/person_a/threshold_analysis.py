"""
threshold_analysis.py - Precision/Recall/Cost-Optimization trade-off analysis across decision thresholds.

COST MODEL & ACTION TIER ASSUMPTIONS:
Production decision_engine.py (src/person_b/decision_engine.py) maps risk scores into
4 tiered operational action bands:
1. Low Risk (score < threshold): "allow"
   - Zero friction; transaction completes instantly without customer friction.
2. Medium Risk (threshold <= score < max(threshold, 0.70)): "flag_for_review"
   - Soft review; transaction goes through seamlessly for the customer while being logged
     for asynchronous analyst review.
   - Legitimate False Positive cost is low (FP_COST_WEIGHT_REVIEW = 0.10, or 10% of transaction value),
     representing internal manual review overhead and analyst triage time.
3. High Risk (max(threshold, 0.70) <= score < 0.85): "hold_for_verification"
   - Step-up authentication (e.g. SMS OTP, card re-auth, 3DS challenge).
   - Legitimate False Positive cost is medium (FP_COST_WEIGHT_HOLD = 0.40, or 40% of transaction value),
     reflecting customer friction, cart abandonment risk, and SMS/auth gateway charges.
4. Very High Risk (score >= 0.85): "auto_decline"
   - Hard block; transaction is rejected outright.
   - Legitimate False Positive cost is 100% of transaction value (FP_COST_WEIGHT_DECLINE = 1.00),
     representing direct lost merchandise revenue and acute customer dissatisfaction.

FALSE NEGATIVES (Undetected Fraud):
- Any fraudulent transaction scoring below threshold lands in "allow".
- Cost = 100% of the fraudulent transaction amount PLUS a fixed chargeback and network penalty
  fee (CHARGEBACK_PENALTY = $500.00), reflecting card network dispute fines, chargeback processing fees,
  and operational recovery overhead.

OPTIMIZATION OBJECTIVE:
- Select the threshold that minimizes total real-world business loss (net_loss = tiered_fp_cost + fn_cost),
  aligning ML operating thresholds directly with dollar impact rather than raw F1.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.person_a.feature_engineering import compute_features_batch, FEATURE_COLUMNS

MODELS_DIR = os.path.join(_PROJECT_ROOT, "data", "models")
PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")
INJECTED_CSV = os.path.join(PROCESSED_DIR, "injected_transactions.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "fraud_model.joblib")
THRESHOLD_JSON = os.path.join(MODELS_DIR, "threshold_analysis.json")

# Friction cost weights per action band for false positives (legitimate transactions flagged)
FP_COST_WEIGHT_REVIEW = 0.10    # 10% friction cost for soft review
FP_COST_WEIGHT_HOLD = 0.40      # 40% friction cost for step-up verification / OTP challenge
FP_COST_WEIGHT_DECLINE = 1.00   # 100% cost (full lost transaction value) for hard declines

# Fixed penalty added per false negative (fraud allowed through) reflecting network dispute/chargeback fees
CHARGEBACK_PENALTY = 500.0


def analyze_thresholds():
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH) or not os.path.exists(INJECTED_CSV):
        print("[threshold_analysis] Model or dataset missing, running training script...")
        from src.person_a.train import train_model
        train_model()

    print(f"[threshold_analysis] Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)

    print(f"[threshold_analysis] Loading dataset from {INJECTED_CSV}...")
    df = pd.read_csv(INJECTED_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    full_df = compute_features_batch(df)

    # Use held-out 20% for threshold analysis
    n_train = int(len(full_df) * 0.8)
    test_df = full_df.iloc[n_train:].copy()

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["is_fraud"].values
    amounts = test_df["amount"].values

    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Evaluate thresholds from 0.05 to 0.99 in fine steps of 0.02 to capture true cost minimum
    threshold_results = []

    for t in np.arange(0.05, 1.00, 0.02):
        t = round(float(t), 2)

        # Tiered action band mapping matching decision_engine.py:
        # - score >= 0.85: auto_decline
        # - score >= max(threshold, 0.70): hold_for_verification
        # - score >= threshold: flag_for_review
        # - else: allow
        auto_decline_mask = (y_pred_prob >= 0.85)
        hold_mask = (y_pred_prob >= max(t, 0.70)) & (~auto_decline_mask)
        review_mask = (y_pred_prob >= t) & (~auto_decline_mask) & (~hold_mask)
        allow_mask = ~(auto_decline_mask | hold_mask | review_mask)

        # Flagged interventions vs unflagged approvals
        flagged_mask = ~allow_mask
        y_pred = flagged_mask.astype(int)

        tp = np.sum((y_pred == 1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        tn = np.sum((y_pred == 0) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Tiered False Positive cost calculation:
        # Legitimate transactions (y_test == 0) categorized by action band friction weight
        fp_review_cost = float(np.sum(amounts[review_mask & (y_test == 0)])) * FP_COST_WEIGHT_REVIEW
        fp_hold_cost = float(np.sum(amounts[hold_mask & (y_test == 0)])) * FP_COST_WEIGHT_HOLD
        fp_decline_cost = float(np.sum(amounts[auto_decline_mask & (y_test == 0)])) * FP_COST_WEIGHT_DECLINE
        tiered_fp_cost = fp_review_cost + fp_hold_cost + fp_decline_cost

        # Fraud caught: monetary value of detected fraud
        fraud_caught = float(np.sum(amounts[flagged_mask & (y_test == 1)]))

        # False Negative cost: undetected fraud allowed through (transaction amount + chargeback penalty)
        fn_mask = allow_mask & (y_test == 1)
        fn_amount_loss = float(np.sum(amounts[fn_mask]))
        fn_cost = fn_amount_loss + (float(fn) * CHARGEBACK_PENALTY)

        # Net loss under real-world cost assumptions
        net_loss = tiered_fp_cost + fn_cost

        threshold_results.append({
            "threshold": t,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "tp_count": int(tp),
            "fp_count": int(fp),
            "fn_count": int(fn),
            "estimated_fp_cost": round(tiered_fp_cost, 2),
            "estimated_fraud_caught": round(fraud_caught, 2),
            "estimated_fn_cost": round(fn_cost, 2),
            "net_loss": round(net_loss, 2)
        })

    # Select optimal threshold by lowest net business loss (cost-optimal rather than F1)
    best_item = min(threshold_results, key=lambda x: x["net_loss"])
    print(f"[threshold_analysis] Optimal Threshold derived: {best_item['threshold']} (Net Loss=${best_item['net_loss']:,.2f}, F1={best_item['f1_score']}, Precision={best_item['precision']}, Recall={best_item['recall']})")

    output_payload = {
        "optimal_threshold": best_item["threshold"],
        "recommended_bands": {
            "low": {"max": best_item["threshold"], "action": "allow"},
            "medium": {"min": best_item["threshold"], "max": max(best_item["threshold"], 0.70), "action": "flag_for_review"},
            "high": {"min": max(best_item["threshold"], 0.70), "max": 0.85, "action": "hold_for_verification"},
            "very_high": {"min": 0.85, "max": 1.00, "action": "auto_decline"}
        },
        "threshold_curve": threshold_results
    }

    with open(THRESHOLD_JSON, "w") as f:
        json.dump(output_payload, f, indent=2)

    print(f"[threshold_analysis] Saved threshold curve report to {THRESHOLD_JSON}")
    return output_payload


def main():
    analyze_thresholds()


if __name__ == "__main__":
    main()

