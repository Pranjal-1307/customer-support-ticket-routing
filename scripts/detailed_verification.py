"""
Detailed Verification Script for M2 Final Verification.
Computes per-class Precision, Recall, and F1-score for Category and Priority
on both the M2 test set and the realistic holdout set.
Identifies category failure cases on realistic holdout.
"""

import os
import sys
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from utils.predict import get_predictor


def run_verification():
    predictor = get_predictor()

    test_path = os.path.join(base_dir, "data", "test.csv")
    holdout_path = os.path.join(base_dir, "data", "evaluation", "m2_realistic_holdout.csv")

    test_df = pd.read_csv(test_path)
    holdout_df = pd.read_csv(holdout_path)

    # 1. Predictions on M2 Test Set
    test_cat_preds, test_pri_preds = [], []
    for _, row in test_df.iterrows():
        res = predictor.predict(row["ticket_text"])
        test_cat_preds.append(res["category"])
        test_pri_preds.append(res["priority"])

    test_df["pred_cat"] = test_cat_preds
    test_df["pred_pri"] = test_pri_preds

    # 2. Predictions on Realistic Holdout
    hold_cat_preds, hold_pri_preds = [], []
    for _, row in holdout_df.iterrows():
        res = predictor.predict(row["ticket_text"])
        hold_cat_preds.append(res["category"])
        hold_pri_preds.append(res["priority"])

    holdout_df["pred_cat"] = hold_cat_preds
    holdout_df["pred_pri"] = hold_pri_preds

    print("=================================================================")
    print("M2 TEST SET — PRIORITY CLASSIFICATION REPORT")
    print("=================================================================")
    print(classification_report(test_df["priority"], test_df["pred_pri"], digits=4))

    print("=================================================================")
    print("REALISTIC HOLDOUT — PRIORITY CLASSIFICATION REPORT")
    print("=================================================================")
    print(classification_report(holdout_df["priority"], holdout_df["pred_pri"], digits=4))

    print("=================================================================")
    print("M2 TEST SET — CATEGORY CLASSIFICATION REPORT")
    print("=================================================================")
    print(classification_report(test_df["category"], test_df["pred_cat"], digits=4))

    print("=================================================================")
    print("REALISTIC HOLDOUT — CATEGORY CLASSIFICATION REPORT")
    print("=================================================================")
    print(classification_report(holdout_df["category"], holdout_df["pred_cat"], digits=4))

    print("=================================================================")
    print("CATEGORY MISCLASSIFICATIONS ON REALISTIC HOLDOUT")
    print("=================================================================")
    mis = holdout_df[holdout_df["category"] != holdout_df["pred_cat"]]
    for _, r in mis.iterrows():
        print(f"Text: '{r['ticket_text']}'")
        print(f"  True Category: {r['category']} | Predicted Category: {r['pred_cat']}")
        print(f"  True Priority: {r['priority']} | Predicted Priority: {r['pred_pri']}\n")


if __name__ == "__main__":
    run_verification()
