"""
Multilingual Semantic Equivalence & Consistency Evaluation for Customer Support Ticket System (M3).
Evaluates cross-lingual semantic consistency across English, Hinglish, and Gujarati.
Calculates:
- Category Accuracy, Macro F1, Per-class F1
- Priority Accuracy & Macro F1
- Multilingual Semantic Consistency Score (%)
- Confusion Matrix
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.predict import get_predictor, predict_ticket
from utils.route import get_department_for_category

# Standard Semantic Equivalence Test Groups across all 8 categories
SEMANTIC_GROUPS = [
    {
        "group_id": "G1_Shipping_Order",
        "expected_category": "Shipping",
        "expected_department": "Shipping / Logistics Team",
        "expected_priority": "Low",
        "items": [
            ("Where is my order?", "en"),
            ("Mera order kaha hai?", "hinglish"),
            ("મારું ઓર્ડર ક્યાં છે?", "gujarati")
        ]
    },
    {
        "group_id": "G2_Shipping_Package",
        "expected_category": "Shipping",
        "expected_department": "Shipping / Logistics Team",
        "expected_priority": "Low",
        "items": [
            ("Where is my package?", "en"),
            ("Mera package kahan hai?", "hinglish"),
            ("મારું પેકેજ ક્યાં છે?", "gujarati")
        ]
    },
    {
        "group_id": "G3_Account_Hacked",
        "expected_category": "Account",
        "expected_department": "Account Support Team",
        "expected_priority": "Critical",
        "items": [
            ("My account has been hacked.", "en"),
            ("Mera account hack ho gaya hai.", "hinglish"),
            ("મારું એકાઉન્ટ હેક થઈ ગયું છે.", "gujarati")
        ]
    },
    {
        "group_id": "G4_Billing_Duplicate",
        "expected_category": "Billing",
        "expected_department": "Billing Team",
        "expected_priority": "Medium",
        "items": [
            ("I was charged twice.", "en"),
            ("Mujhse do baar payment li gayi.", "hinglish"),
            ("મારા પાસેથી બે વાર પૈસા લેવામાં આવ્યા છે.", "gujarati")
        ]
    },
    {
        "group_id": "G5_Refund_Request",
        "expected_category": "Refund",
        "expected_department": "Refund Team",
        "expected_priority": "Medium",
        "items": [
            ("I want a refund for my order.", "en"),
            ("Mujhe mere order ka refund chahiye.", "hinglish"),
            ("મને મારા ઓર્ડરનું રિફંડ જોઈએ છે.", "gujarati")
        ]
    },
    {
        "group_id": "G6_TechSupport_Crash",
        "expected_category": "Technical Support",
        "expected_department": "Technical Support Team",
        "expected_priority": "Medium",
        "items": [
            ("The mobile app keeps crashing.", "en"),
            ("App baar baar crash ho raha hai.", "hinglish"),
            ("એપ વારંવાર ક્રેશ થઈ રહી છે.", "gujarati")
        ]
    },
    {
        "group_id": "G7_Complaint_Service",
        "expected_category": "Complaint",
        "expected_department": "Customer Care Team",
        "expected_priority": "Low",
        "items": [
            ("Worst customer service.", "en"),
            ("Bohot hi kharab customer service hai.", "hinglish"),
            ("ખૂબ જ ખરાબ ગ્રાહક સેવા છે.", "gujarati")
        ]
    },
    {
        "group_id": "G8_Product_DarkMode",
        "expected_category": "Product Inquiry",
        "expected_department": "Sales / General Inquiry Team",
        "expected_priority": "Low",
        "items": [
            ("Does product support dark mode?", "en"),
            ("Kya is product mein dark mode support hai?", "hinglish"),
            ("શું આ પ્રોડક્ટ ડાર્ક મોડને સપોર્ટ કરે છે?", "gujarati")
        ]
    },
    {
        "group_id": "G9_Cancellation_Subscription",
        "expected_category": "Cancellation",
        "expected_department": "Retention / Cancellation Team",
        "expected_priority": "Medium",
        "items": [
            ("I want to cancel my subscription.", "en"),
            ("Mujhe apna subscription cancel karna hai.", "hinglish"),
            ("મારે મારું સબ્સ્ક્રિપ્શન રદ કરવું છે.", "gujarati")
        ]
    }
]


def evaluate_multilingual_consistency():
    predictor = get_predictor()
    results = []
    consistent_groups = 0
    total_groups = len(SEMANTIC_GROUPS)

    print("=" * 80)
    print("M3 MULTILINGUAL SEMANTIC CONSISTENCY EVALUATION")
    print("=" * 80)

    for g in SEMANTIC_GROUPS:
        gid = g["group_id"]
        exp_cat = g["expected_category"]
        exp_dept = g["expected_department"]
        
        preds = []
        print(f"\n--- Group: {gid} (Expected Category: {exp_cat}, Dept: {exp_dept}) ---")
        
        for text, lang in g["items"]:
            pred = predictor.predict(text)
            pred_cat = pred["category"]
            pred_dept = pred["department"]
            pred_pri = pred["priority"]
            conf = pred["category_confidence"]
            preds.append((text, lang, pred_cat, pred_dept, pred_pri, conf))
            
            is_correct = (pred_cat == exp_cat)
            mark = "[OK]" if is_correct else "[FAIL]"
            print(f"  {mark} [{lang:<8}] {text:<42} -> Cat: {pred_cat:<18} (Conf: {conf:.1f}%) | Dept: {pred_dept}")

        # Check if all items in this group received the same category
        categories_in_group = set(p[2] for p in preds)
        all_same = (len(categories_in_group) == 1)
        all_correct = all_same and (list(categories_in_group)[0] == exp_cat)

        if all_correct:
            consistent_groups += 1
            print(f"  --> Group Status: PERFECT CONSISTENCY (100% Match)")
        elif all_same:
            print(f"  --> Group Status: CONSISTENT BUT INCORRECT ({list(categories_in_group)[0]})")
        else:
            print(f"  --> Group Status: INCONSISTENT (Spread: {categories_in_group})")

        results.append({
            "group_id": gid,
            "expected_category": exp_cat,
            "predictions": preds,
            "is_consistent": all_same,
            "is_correct": all_correct
        })

    consistency_score = (consistent_groups / total_groups) * 100.0
    print("\n" + "=" * 80)
    print(f"MULTILINGUAL SEMANTIC CONSISTENCY SCORE: {consistency_score:.2f}% ({consistent_groups}/{total_groups} Groups)")
    print("=" * 80)

    return consistency_score, results


def evaluate_test_set_performance():
    test_path = os.path.join(BASE_DIR, "data", "test.csv")
    if not os.path.exists(test_path):
        test_path = os.path.join(BASE_DIR, "dataset", "test.csv")

    test_df = pd.read_csv(test_path)
    predictor = get_predictor()

    y_true_cat = test_df["category"].tolist()
    y_true_pri = test_df["priority"].tolist()

    y_pred_cat = []
    y_pred_pri = []

    for text in test_df["ticket_text"]:
        p = predictor.predict(text)
        y_pred_cat.append(p["category"])
        y_pred_pri.append(p["priority"])

    cat_acc = accuracy_score(y_true_cat, y_pred_cat)
    cat_prec, cat_rec, cat_f1, _ = precision_recall_fscore_support(y_true_cat, y_pred_cat, average='weighted', zero_division=0)
    _, _, cat_macro_f1, _ = precision_recall_fscore_support(y_true_cat, y_pred_cat, average='macro', zero_division=0)

    pri_acc = accuracy_score(y_true_pri, y_pred_pri)
    pri_prec, pri_rec, pri_f1, _ = precision_recall_fscore_support(y_true_pri, y_pred_pri, average='weighted', zero_division=0)
    _, _, pri_macro_f1, _ = precision_recall_fscore_support(y_true_pri, y_pred_pri, average='macro', zero_division=0)

    print("\n" + "=" * 80)
    print("M3 OVERALL TEST SET PERFORMANCE EVALUATION")
    print("=" * 80)
    print(f"Category Accuracy:     {cat_acc:.4f}")
    print(f"Category Weighted F1:  {cat_f1:.4f}")
    print(f"Category Macro F1:     {cat_macro_f1:.4f}")
    print("-" * 40)
    print(f"Priority Accuracy:     {pri_acc:.4f}")
    print(f"Priority Weighted F1:  {pri_f1:.4f}")
    print(f"Priority Macro F1:     {pri_macro_f1:.4f}")
    print("=" * 80)

    print("\n--- CATEGORY CLASSIFICATION REPORT ---")
    print(classification_report(y_true_cat, y_pred_cat, zero_division=0))

    print("\n--- CATEGORY CONFUSION MATRIX ---")
    labels = sorted(list(set(y_true_cat)))
    cm = confusion_matrix(y_true_cat, y_pred_cat, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)

    return {
        "category_accuracy": round(float(cat_acc), 4),
        "category_macro_f1": round(float(cat_macro_f1), 4),
        "category_weighted_f1": round(float(cat_f1), 4),
        "priority_accuracy": round(float(pri_acc), 4),
        "priority_macro_f1": round(float(pri_macro_f1), 4),
        "priority_weighted_f1": round(float(pri_f1), 4),
    }


if __name__ == "__main__":
    evaluate_multilingual_consistency()
    evaluate_test_set_performance()
