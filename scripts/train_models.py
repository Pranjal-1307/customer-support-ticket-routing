"""
Model Training and Evaluation Script for Customer Support Ticket System (Milestone 2).
Compairs Logistic Regression, Naive Bayes, Random Forest, and Calibrated Linear SVM
for both Category and Priority classification.
Selects the best classifiers based on Weighted F1-score and saves serialized artifacts.
Outputs detailed metrics (Accuracy, Precision, Recall, Weighted F1, Macro F1).
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

from utils.preprocess import preprocess_text


def train_and_evaluate_models():
    dataset_dir = os.path.join(base_dir, "data")
    legacy_dir = os.path.join(base_dir, "dataset")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    train_path = os.path.join(dataset_dir, "train.csv")
    test_path = os.path.join(dataset_dir, "test.csv")

    if not os.path.exists(train_path):
        train_path = os.path.join(legacy_dir, "train.csv")
        test_path = os.path.join(legacy_dir, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Train/Test datasets not found. Run generate_data.py first.")

    print("Loading M2 datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("Preprocessing text data...")
    train_df['processed_text'] = train_df['ticket_text'].apply(preprocess_text)
    test_df['processed_text'] = test_df['ticket_text'].apply(preprocess_text)

    # Fallback for empty text
    train_df['processed_text'] = train_df['processed_text'].replace("", "ticket")
    test_df['processed_text'] = test_df['processed_text'].replace("", "ticket")

    # TF-IDF Feature Extraction (Fitted ONLY on train_df)
    print("Extracting TF-IDF features (Fitted strictly on Train set)...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df['processed_text'])
    X_test = vectorizer.transform(test_df['processed_text'])

    # Category Targets
    label_encoder = LabelEncoder()
    y_train_cat = label_encoder.fit_transform(train_df['category'])
    y_test_cat = label_encoder.transform(test_df['category'])

    # Priority Targets
    y_train_pri = train_df['priority']
    y_test_pri = test_df['priority']

    # Candidate Models definition
    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Calibrated Linear SVM": CalibratedClassifierCV(estimator=LinearSVC(random_state=42), cv=5)
    }

    category_metrics = {}
    best_cat_model_name = None
    best_cat_f1 = -1.0
    best_cat_clf = None

    print("\n--- CATEGORY CLASSIFICATION EVALUATION ---")
    for name, clf in candidate_models.items():
        print(f"Training {name} for Category...")
        clf.fit(X_train, y_train_cat)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test_cat, y_pred)
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_test_cat, y_pred, average='weighted', zero_division=0)
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_test_cat, y_pred, average='macro', zero_division=0)

        category_metrics[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec_w), 4),
            "recall": round(float(rec_w), 4),
            "f1_score": round(float(f1_w), 4),
            "macro_f1": round(float(f1_m), 4),
            "weighted_f1": round(float(f1_w), 4)
        }

        print(f"  {name:<25} -> Acc: {acc:.4f}, Weighted F1: {f1_w:.4f}, Macro F1: {f1_m:.4f}")

        if f1_w > best_cat_f1:
            best_cat_f1 = f1_w
            best_cat_model_name = name
            best_cat_clf = clf

    print(f"\n>>> Best Category Model: {best_cat_model_name} (Weighted F1: {best_cat_f1:.4f}) <<<")

    # Priority Evaluation across candidate models
    priority_metrics = {}
    best_pri_model_name = None
    best_pri_f1 = -1.0
    best_pri_clf = None

    print("\n--- PRIORITY CLASSIFICATION EVALUATION ---")
    pri_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "Calibrated Linear SVM": CalibratedClassifierCV(estimator=LinearSVC(class_weight='balanced', random_state=42), cv=5)
    }

    for name, clf in pri_models.items():
        print(f"Training {name} for Priority...")
        clf.fit(X_train, y_train_pri)
        y_pred_pri = clf.predict(X_test)

        acc = accuracy_score(y_test_pri, y_pred_pri)
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_test_pri, y_pred_pri, average='weighted', zero_division=0)
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_test_pri, y_pred_pri, average='macro', zero_division=0)

        priority_metrics[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec_w), 4),
            "recall": round(float(rec_w), 4),
            "f1_score": round(float(f1_w), 4),
            "macro_f1": round(float(f1_m), 4),
            "weighted_f1": round(float(f1_w), 4)
        }

        print(f"  {name:<25} -> Acc: {acc:.4f}, Weighted F1: {f1_w:.4f}, Macro F1: {f1_m:.4f}")

        if f1_w > best_pri_f1:
            best_pri_f1 = f1_w
            best_pri_model_name = name
            best_pri_clf = clf

    print(f"\n>>> Best Priority Model: {best_pri_model_name} (Weighted F1: {best_pri_f1:.4f}) <<<")

    # Save artifacts
    print("\nSaving model artifacts to models/ directory...")
    joblib.dump(vectorizer, os.path.join(models_dir, "vectorizer.pkl"))
    joblib.dump(label_encoder, os.path.join(models_dir, "label_encoder.pkl"))
    joblib.dump(best_cat_clf, os.path.join(models_dir, "ticket_classifier.pkl"))
    joblib.dump(best_pri_clf, os.path.join(models_dir, "priority_classifier.pkl"))

    summary = {
        "best_category_model": best_cat_model_name,
        "best_category_f1": round(float(best_cat_f1), 4),
        "best_priority_model": best_pri_model_name,
        "best_priority_f1": round(float(best_pri_f1), 4),
        "category_metrics": category_metrics,
        "priority_metrics": priority_metrics
    }

    metrics_path = os.path.join(models_dir, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Saved model metrics to {metrics_path}")
    return summary


if __name__ == "__main__":
    train_and_evaluate_models()
