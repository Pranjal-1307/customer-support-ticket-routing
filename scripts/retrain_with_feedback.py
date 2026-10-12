"""
Phase 12: Validation Gate & Controlled Retraining with Human Feedback (M3).

Workflow:
1. Load Base Training Data (data/train.csv)
2. Fetch ONLY Validated Feedback (validated = 1) from SQLite (tickets.db)
3. Build Candidate Training Dataset
4. Train Candidate Model (TF-IDF + 6 Priority Signals)
5. Evaluate on Validation Set and Fixed Holdout Set
6. Compare Metrics against Active Production Model
7. Output Explicit Promotion Decision & Gate Recommendation
8. Promote ONLY if explicit '--promote' flag is supplied and promotion criteria are met.
"""

import os
import sys
import json
import sqlite3
import argparse
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.sparse import hstack, csr_matrix
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.preprocess import preprocess_text
from utils.priority_signals import extract_priority_signals, signals_to_feature_vector

DB_PATH = os.path.join(BASE_DIR, "tickets.db")


def get_validated_feedback():
    """Queries SQLite database for feedback rows that have passed the validation gate."""
    if not os.path.exists(DB_PATH):
        return []
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT ticket_text, corrected_priority FROM priority_feedback WHERE validated = 1")
        rows = cursor.fetchall()
        feedback = [{"ticket_text": r["ticket_text"], "priority": r["corrected_priority"]} for r in rows if r["ticket_text"]]
    except Exception as e:
        print(f"Notice: Could not read feedback table ({e}). Returning 0 feedback entries.")
        feedback = []
    finally:
        conn.close()
    return feedback


def run_controlled_retraining(promote: bool = False):
    print("=" * 80)
    print("M3 CONTROLLED RETRAINING & VALIDATION GATE")
    print("=" * 80)

    # 1. Load Base Training Data
    train_df = pd.read_csv(os.path.join(BASE_DIR, "data", "train.csv"))
    val_df = pd.read_csv(os.path.join(BASE_DIR, "data", "validation.csv"))
    holdout_df = pd.read_csv(os.path.join(BASE_DIR, "data", "evaluation", "m2_realistic_holdout.csv"))

    # 2. Fetch Validated Feedback
    validated_feedback = get_validated_feedback()
    print(f"Base Training Samples: {len(train_df)}")
    print(f"Validated Human Feedback Samples Found: {len(validated_feedback)}")

    # 3. Build Candidate Training Data
    candidate_train_df = train_df[['ticket_text', 'priority']].copy()
    if validated_feedback:
        fb_df = pd.DataFrame(validated_feedback)
        candidate_train_df = pd.concat([candidate_train_df, fb_df], ignore_index=True)
        print(f"Combined Candidate Training Size: {len(candidate_train_df)}")

    # Preprocess & Vectorize
    vectorizer = joblib.load(os.path.join(BASE_DIR, "models", "vectorizer.pkl"))

    X_train_clean = [preprocess_text(t) or "ticket" for t in candidate_train_df['ticket_text']]
    y_train = candidate_train_df['priority'].values

    X_val_clean = [preprocess_text(t) or "ticket" for t in val_df['ticket_text']]
    y_val = val_df['priority'].values

    X_holdout_clean = [preprocess_text(t) or "ticket" for t in holdout_df['ticket_text']]
    y_holdout = holdout_df['priority'].values

    # TF-IDF Features
    X_train_tfidf = vectorizer.transform(X_train_clean)
    X_val_tfidf = vectorizer.transform(X_val_clean)
    X_holdout_tfidf = vectorizer.transform(X_holdout_clean)

    # Priority Signal Features
    def extract_sig_mat(texts):
        vecs = [signals_to_feature_vector(extract_priority_signals(t)) for t in texts]
        return csr_matrix(np.array(vecs, dtype=np.float32))

    X_train_sig = extract_sig_mat(candidate_train_df['ticket_text'])
    X_val_sig = extract_sig_mat(val_df['ticket_text'])
    X_holdout_sig = extract_sig_mat(holdout_df['ticket_text'])

    # Stacking
    X_train_comb = hstack([X_train_tfidf, X_train_sig]).tocsr()
    X_val_comb = hstack([X_val_tfidf, X_val_sig]).tocsr()
    X_holdout_comb = hstack([X_holdout_tfidf, X_holdout_sig]).tocsr()

    # 4. Train Candidate Model
    candidate_model = CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced', random_state=42), cv=3)
    candidate_model.fit(X_train_comb, y_train)

    # 5. Evaluate Candidate Model
    val_pred = candidate_model.predict(X_val_comb)
    val_acc = accuracy_score(y_val, val_pred)
    _, _, val_macro_f1, _ = precision_recall_fscore_support(y_val, val_pred, average='macro', zero_division=0)

    holdout_pred = candidate_model.predict(X_holdout_comb)
    holdout_acc = accuracy_score(y_holdout, holdout_pred)
    _, _, holdout_macro_f1, _ = precision_recall_fscore_support(y_holdout, holdout_pred, average='macro', zero_division=0)

    # 6. Evaluate Active Production Model on Holdout
    prod_model_path = os.path.join(BASE_DIR, "models", "priority_classifier.pkl")
    prod_model = joblib.load(prod_model_path)
    prod_val_pred = prod_model.predict(X_val_tfidf)
    prod_val_acc = accuracy_score(y_val, prod_val_pred)
    prod_holdout_pred = prod_model.predict(X_holdout_tfidf)
    prod_holdout_acc = accuracy_score(y_holdout, prod_holdout_pred)
    _, _, prod_holdout_macro_f1, _ = precision_recall_fscore_support(y_holdout, prod_holdout_pred, average='macro', zero_division=0)

    print("\n--- PERFORMANCE EVALUATION ---")
    print(f"Production Model -> Val Acc: {prod_val_acc*100:.2f}% | Holdout Acc: {prod_holdout_acc*100:.2f}% | Holdout Macro F1: {prod_holdout_macro_f1:.4f}")
    print(f"Candidate Model  -> Val Acc: {val_acc*100:.2f}% | Holdout Acc: {holdout_acc*100:.2f}% | Holdout Macro F1: {holdout_macro_f1:.4f}")

    # 7. Candidate Model Artifact Persistence (Isolated)
    candidate_artifact_path = os.path.join(BASE_DIR, "models", "m3_priority_classifier_candidate.pkl")
    joblib.dump(candidate_model, candidate_artifact_path)
    print(f"\nCandidate model saved to: {candidate_artifact_path}")

    # 8. Promotion Gate Decision
    is_promotable = (holdout_acc >= prod_holdout_acc) and (val_acc >= 0.85)

    gate_decision = {
        "timestamp": datetime.now().isoformat(),
        "candidate_val_accuracy": round(val_acc, 4),
        "candidate_val_macro_f1": round(val_macro_f1, 4),
        "candidate_holdout_accuracy": round(holdout_acc, 4),
        "candidate_holdout_macro_f1": round(holdout_macro_f1, 4),
        "production_holdout_accuracy": round(prod_holdout_acc, 4),
        "production_holdout_macro_f1": round(prod_holdout_macro_f1, 4),
        "validated_feedback_samples_used": len(validated_feedback),
        "promotion_gate_passed": is_promotable,
        "explicit_promotion_requested": promote,
        "action_taken": "Promoted to production" if (promote and is_promotable) else "Retained candidate in isolation"
    }

    print("\n--- VALIDATION GATE DECISION ---")
    if is_promotable:
        print("[PASS] Candidate meets or exceeds production quality criteria.")
    else:
        print("[HOLD] Candidate did not meet automated promotion threshold. Retaining active production model.")

    if promote:
        if is_promotable:
            # Backup current production to v1
            v1_path = os.path.join(BASE_DIR, "models", "priority_classifier_v1.pkl")
            if not os.path.exists(v1_path):
                joblib.dump(prod_model, v1_path)
            # Save candidate as v2 and active production
            v2_path = os.path.join(BASE_DIR, "models", "priority_classifier_v2.pkl")
            joblib.dump(candidate_model, v2_path)
            joblib.dump(candidate_model, prod_model_path)
            print("[PROMOTION COMPLETE] Production model updated to candidate v2.0.")
        else:
            print("[PROMOTION ABORTED] Explicit promotion rejected by validation gate.")
    else:
        print("[INFO] Production model untouched. (Pass '--promote' to execute promotion gate if passed).")

    return gate_decision

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M3 Controlled Retraining & Validation Gate")
    parser.add_argument("--promote", action="store_true", help="Explicitly promote candidate to production if validation gate passes.")
    args = parser.parse_args()
    run_controlled_retraining(promote=args.promote)
