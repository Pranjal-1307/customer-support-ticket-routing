"""
Model Training and Evaluation Script for Customer Support Ticket System.
Compares Logistic Regression, Naive Bayes, Random Forest, and Linear SVM for Category classification.
Selects the best classifier based on F1-score and trains Priority classifier.
Saves serialized models and vectorizer artifacts.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

from utils.preprocess import preprocess_text


def train_and_evaluate_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, "dataset")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    train_path = os.path.join(dataset_dir, "train.csv")
    test_path = os.path.join(dataset_dir, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Train/Test datasets not found. Run generate_data.py first.")

    print("Loading datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("Preprocessing text data...")
    train_df['processed_text'] = train_df['ticket_text'].apply(preprocess_text)
    test_df['processed_text'] = test_df['ticket_text'].apply(preprocess_text)

    # Clean missing/empty processed text fallback
    train_df['processed_text'] = train_df['processed_text'].replace("", "ticket")
    test_df['processed_text'] = test_df['processed_text'].replace("", "ticket")

    # TF-IDF Feature Extraction
    print("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df['processed_text'])
    X_test = vectorizer.transform(test_df['processed_text'])

    # Category Label Encoding
    label_encoder = LabelEncoder()
    y_train_cat = label_encoder.fit_transform(train_df['category'])
    y_test_cat = label_encoder.transform(test_df['category'])

    # Priority Targets
    y_train_pri = train_df['priority']
    y_test_pri = test_df['priority']

    # Define Candidate Models for Category Classification
    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Linear SVM": LinearSVC(random_state=42)
    }

    model_metrics = {}
    best_model_name = None
    best_f1_score = -1.0
    best_category_model = None

    print("\n--- Category Classification Models Evaluation ---")
    for name, clf in candidate_models.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train, y_train_cat)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test_cat, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test_cat, y_pred, average='weighted')
        report = classification_report(y_test_cat, y_pred, target_names=label_encoder.classes_, output_dict=True)
        cm = confusion_matrix(y_test_cat, y_pred).tolist()

        model_metrics[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "classification_report": report,
            "confusion_matrix": cm
        }

        print(f"{name} -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, Weighted F1: {f1:.4f}")

        if f1 > best_f1_score:
            best_f1_score = f1
            best_model_name = name
            best_category_model = clf

    print(f"\n>>> Selected Best Model: {best_model_name} with Weighted F1-Score: {best_f1_score:.4f} <<<")

    # Priority Classifier Training (Logistic Regression)
    print("\nTraining Priority Classifier (Logistic Regression)...")
    priority_classifier = LogisticRegression(max_iter=1000, random_state=42)
    priority_classifier.fit(X_train, y_train_pri)
    y_pred_pri = priority_classifier.predict(X_test)
    pri_acc = accuracy_score(y_test_pri, y_pred_pri)
    pri_prec, pri_rec, pri_f1, _ = precision_recall_fscore_support(y_test_pri, y_pred_pri, average='weighted')

    print(f"Priority Classifier -> Accuracy: {pri_acc:.4f}, F1-Score: {pri_f1:.4f}")

    # Save Models and Artifacts
    print("\nSaving model artifacts to models/ directory...")
    joblib.dump(vectorizer, os.path.join(models_dir, "vectorizer.pkl"))
    joblib.dump(label_encoder, os.path.join(models_dir, "label_encoder.pkl"))
    joblib.dump(best_category_model, os.path.join(models_dir, "ticket_classifier.pkl"))
    joblib.dump(priority_classifier, os.path.join(models_dir, "priority_classifier.pkl"))

    summary_metrics = {
        "best_category_model_name": best_model_name,
        "best_category_f1": round(float(best_f1_score), 4),
        "priority_accuracy": round(float(pri_acc), 4),
        "priority_f1": round(float(pri_f1), 4),
        "models_comparison": model_metrics
    }

    metrics_path = os.path.join(models_dir, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary_metrics, f, indent=4)

    print(f"Saved evaluation metrics to {metrics_path}")
    print("Model training & serialization complete!")
    return summary_metrics


if __name__ == "__main__":
    train_and_evaluate_models()
