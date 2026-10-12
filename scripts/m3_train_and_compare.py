"""
Phase 5 & 6: Train and Fairly Compare Priority Classification Models for M3.
Compares:
1. M2 Baseline (TF-IDF only)
2. Logistic Regression (TF-IDF only vs TF-IDF + Priority Signals)
3. Linear SVM / Calibrated Linear SVM (TF-IDF only vs TF-IDF + Priority Signals)
4. Random Forest (TF-IDF only vs TF-IDF + Priority Signals)

Evaluates on:
- Synthetic Test Set (data/test.csv)
- Realistic Holdout Set (data/evaluation/m2_realistic_holdout.csv)

Saves candidate model to:
models/m3_priority_classifier_candidate.pkl
(Preserves models/priority_classifier.pkl untouched)
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.preprocess import preprocess_text
from utils.priority_signals import extract_priority_signals, signals_to_feature_vector


def extract_signal_matrix(texts: list) -> csr_matrix:
    signals_list = [signals_to_feature_vector(extract_priority_signals(t)) for t in texts]
    return csr_matrix(np.array(signals_list, dtype=np.float32))


def train_and_compare_m3():
    # 1. Load Data
    train_df = pd.read_csv(os.path.join(BASE_DIR, "data", "train.csv"))
    val_df = pd.read_csv(os.path.join(BASE_DIR, "data", "validation.csv"))
    test_df = pd.read_csv(os.path.join(BASE_DIR, "data", "test.csv"))
    holdout_df = pd.read_csv(os.path.join(BASE_DIR, "data", "evaluation", "m2_realistic_holdout.csv"))

    # Preprocess texts
    X_train_raw = train_df['ticket_text'].tolist()
    X_train_clean = [preprocess_text(t) or "ticket" for t in X_train_raw]
    y_train = train_df['priority'].values

    X_test_raw = test_df['ticket_text'].tolist()
    X_test_clean = [preprocess_text(t) or "ticket" for t in X_test_raw]
    y_test = test_df['priority'].values

    X_holdout_raw = holdout_df['ticket_text'].tolist()
    X_holdout_clean = [preprocess_text(t) or "ticket" for t in X_holdout_raw]
    y_holdout = holdout_df['priority'].values

    # Load frozen M2 vectorizer or fit matching vectorizer
    vec_path = os.path.join(BASE_DIR, "models", "vectorizer.pkl")
    vectorizer = joblib.load(vec_path)

    X_train_tfidf = vectorizer.transform(X_train_clean)
    X_test_tfidf = vectorizer.transform(X_test_clean)
    X_holdout_tfidf = vectorizer.transform(X_holdout_clean)

    # Compute Signal Features
    X_train_sig = extract_signal_matrix(X_train_raw)
    X_test_sig = extract_signal_matrix(X_test_raw)
    X_holdout_sig = extract_signal_matrix(X_holdout_raw)

    # Combined Features: TF-IDF + Priority Signals
    X_train_comb = hstack([X_train_tfidf, X_train_sig]).tocsr()
    X_test_comb = hstack([X_test_tfidf, X_test_sig]).tocsr()
    X_holdout_comb = hstack([X_holdout_tfidf, X_holdout_sig]).tocsr()

    classes = ['Critical', 'High', 'Low', 'Medium']

    # Define Models to Compare
    models = {
        "M2 Baseline (Calibrated Linear SVM - TFIDF only)": {
            "model": CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42), cv=3),
            "X_train": X_train_tfidf, "X_test": X_test_tfidf, "X_holdout": X_holdout_tfidf, "is_comb": False
        },
        "Logistic Regression (TF-IDF only)": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "X_train": X_train_tfidf, "X_test": X_test_tfidf, "X_holdout": X_holdout_tfidf, "is_comb": False
        },
        "Random Forest (TF-IDF only)": {
            "model": RandomForestClassifier(n_estimators=100, random_state=42),
            "X_train": X_train_tfidf, "X_test": X_test_tfidf, "X_holdout": X_holdout_tfidf, "is_comb": False
        },
        "Logistic Regression (TF-IDF + Signals)": {
            "model": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
            "X_train": X_train_comb, "X_test": X_test_comb, "X_holdout": X_holdout_comb, "is_comb": True
        },
        "Random Forest (TF-IDF + Signals)": {
            "model": RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42),
            "X_train": X_train_comb, "X_test": X_test_comb, "X_holdout": X_holdout_comb, "is_comb": True
        },
        "Calibrated Linear SVM (TF-IDF + Signals) [M3 Candidate]": {
            "model": CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced', random_state=42), cv=3),
            "X_train": X_train_comb, "X_test": X_test_comb, "X_holdout": X_holdout_comb, "is_comb": True
        }
    }

    comparison_results = {}

    print("=" * 80)
    print("M3 PRIORITY MODEL COMPARISON BENCHMARK")
    print("=" * 80)

    for name, config in models.items():
        clf = config["model"]
        clf.fit(config["X_train"], y_train)

        # Test evaluation
        y_test_pred = clf.predict(config["X_test"])
        test_acc = accuracy_score(y_test, y_test_pred)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_test_pred, average='macro', zero_division=0)
        p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_test, y_test_pred, average='weighted', zero_division=0)
        
        # Per class metrics on Test
        p_cls, r_cls, f_cls, s_cls = precision_recall_fscore_support(y_test, y_test_pred, labels=classes, zero_division=0)
        test_cm = confusion_matrix(y_test, y_test_pred, labels=classes).tolist()

        # Realistic Holdout evaluation
        y_holdout_pred = clf.predict(config["X_holdout"])
        holdout_acc = accuracy_score(y_holdout, y_holdout_pred)
        hp_macro, hr_macro, hf1_macro, _ = precision_recall_fscore_support(y_holdout, y_holdout_pred, average='macro', zero_division=0)
        hp_wt, hr_wt, hf1_wt, _ = precision_recall_fscore_support(y_holdout, y_holdout_pred, average='weighted', zero_division=0)
        hp_cls, hr_cls, hf_cls, _ = precision_recall_fscore_support(y_holdout, y_holdout_pred, labels=classes, zero_division=0)
        holdout_cm = confusion_matrix(y_holdout, y_holdout_pred, labels=classes).tolist()

        comparison_results[name] = {
            "test_accuracy": round(test_acc, 4),
            "test_macro_f1": round(f1_macro, 4),
            "test_weighted_f1": round(f1_wt, 4),
            "test_per_class": {
                cls: {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}
                for cls, p, r, f in zip(classes, p_cls, r_cls, f_cls)
            },
            "test_confusion_matrix": test_cm,
            "holdout_accuracy": round(holdout_acc, 4),
            "holdout_macro_f1": round(hf1_macro, 4),
            "holdout_weighted_f1": round(hf1_wt, 4),
            "holdout_per_class": {
                cls: {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}
                for cls, p, r, f in zip(classes, hp_cls, hr_cls, hf_cls)
            },
            "holdout_confusion_matrix": holdout_cm
        }

        print(f"\nModel: {name}")
        print(f"  Synthetic Test -> Acc: {test_acc:.4f} | Macro F1: {f1_macro:.4f} | Weighted F1: {f1_wt:.4f}")
        print(f"  Realistic Holdout -> Acc: {holdout_acc:.4f} | Macro F1: {hf1_macro:.4f} | Weighted F1: {hf1_wt:.4f}")
        print(f"  Holdout Recalls: Critical={hr_cls[0]:.2f}, High={hr_cls[1]:.2f}, Low={hr_cls[2]:.2f}, Medium={hr_cls[3]:.2f}")

    # Save candidate model artifact
    candidate_model = models["Calibrated Linear SVM (TF-IDF + Signals) [M3 Candidate]"]["model"]
    candidate_path = os.path.join(BASE_DIR, "models", "m3_priority_classifier_candidate.pkl")
    joblib.dump(candidate_model, candidate_path)
    print(f"\nSaved M3 Candidate Model to: {candidate_path}")
    print(f"Preserved Production Model at: {os.path.join(BASE_DIR, 'models', 'priority_classifier.pkl')}")

    # Save benchmark results JSON
    benchmark_path = os.path.join(BASE_DIR, "models", "m3_model_comparison.json")
    with open(benchmark_path, "w") as f:
        json.dump(comparison_results, f, indent=4)

    return comparison_results

if __name__ == "__main__":
    train_and_compare_m3()
