"""
Prediction Module for Customer Support Ticket Routing System.
Loads trained model artifacts, preprocesses raw ticket text, predicts category and priority,
calculates model confidence, and routes to appropriate department.
"""

import os
import joblib
import numpy as np
from utils.preprocess import preprocess_text
from utils.route import get_department_for_category


class TicketPredictor:
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, "models")

        self.models_dir = models_dir
        self.vectorizer = None
        self.label_encoder = None
        self.ticket_classifier = None
        self.priority_classifier = None
        self.is_loaded = False
        self.load_models()

    def load_models(self):
        """Loads all serialized model artifacts from models directory."""
        try:
            vec_path = os.path.join(self.models_dir, "vectorizer.pkl")
            encoder_path = os.path.join(self.models_dir, "label_encoder.pkl")
            classifier_path = os.path.join(self.models_dir, "ticket_classifier.pkl")
            priority_path = os.path.join(self.models_dir, "priority_classifier.pkl")

            if not all(os.path.exists(p) for p in [vec_path, encoder_path, classifier_path, priority_path]):
                self.is_loaded = False
                return

            self.vectorizer = joblib.load(vec_path)
            self.label_encoder = joblib.load(encoder_path)
            self.ticket_classifier = joblib.load(classifier_path)
            self.priority_classifier = joblib.load(priority_path)
            self.is_loaded = True
        except Exception as e:
            self.is_loaded = False
            print(f"Error loading model artifacts: {e}")

    def predict(self, ticket_text: str) -> dict:
        """
        Processes raw ticket text and returns predicted category, priority, department, confidence, and status.
        """
        if not self.is_loaded:
            self.load_models()
            if not self.is_loaded:
                return {
                    "status": "error",
                    "error": "Model artifacts not found or failed to load. Please train models first.",
                    "category": "Unclassified",
                    "priority": "Medium",
                    "department": "General Support Team",
                    "confidence": 0.0
                }

        if not isinstance(ticket_text, str) or not ticket_text.strip():
            return {
                "status": "error",
                "error": "Input ticket text cannot be empty.",
                "category": "Unclassified",
                "priority": "Medium",
                "department": "General Support Team",
                "confidence": 0.0
            }

        try:
            # 1. Preprocess
            processed = preprocess_text(ticket_text)
            if not processed:
                processed = "ticket"

            # 2. Extract Features
            vec_input = self.vectorizer.transform([processed])

            # 3. Category Prediction & Confidence Calculation
            if hasattr(self.ticket_classifier, "predict_proba"):
                probs = self.ticket_classifier.predict_proba(vec_input)[0]
                pred_idx = np.argmax(probs)
                cat_confidence = float(probs[pred_idx])
            elif hasattr(self.ticket_classifier, "decision_function"):
                dec = self.ticket_classifier.decision_function(vec_input)[0]
                exp_dec = np.exp(dec - np.max(dec))
                probs = exp_dec / exp_dec.sum()
                pred_idx = np.argmax(probs)
                cat_confidence = float(probs[pred_idx])
            else:
                pred_idx = self.ticket_classifier.predict(vec_input)[0]
                cat_confidence = 1.0

            predicted_category = str(self.label_encoder.inverse_transform([pred_idx])[0])

            # 4. Priority Prediction & Confidence Calculation
            if hasattr(self.priority_classifier, "predict_proba"):
                pri_probs = self.priority_classifier.predict_proba(vec_input)[0]
                pri_idx = np.argmax(pri_probs)
                predicted_priority = str(self.priority_classifier.classes_[pri_idx])
                pri_confidence = float(pri_probs[pri_idx])
            elif hasattr(self.priority_classifier, "decision_function"):
                pri_dec = self.priority_classifier.decision_function(vec_input)[0]
                exp_dec = np.exp(pri_dec - np.max(pri_dec))
                pri_probs = exp_dec / exp_dec.sum()
                pri_idx = np.argmax(pri_probs)
                predicted_priority = str(self.priority_classifier.classes_[pri_idx])
                pri_confidence = float(pri_probs[pri_idx])
            else:
                predicted_priority = str(self.priority_classifier.predict(vec_input)[0])
                pri_confidence = 1.0

            # 5. Department Routing
            department = get_department_for_category(predicted_category)

            cat_conf_pct = round(cat_confidence * 100, 2)
            pri_conf_pct = round(pri_confidence * 100, 2)

            return {
                "status": "success",
                "ticket_text": ticket_text,
                "category": predicted_category,
                "category_confidence": cat_conf_pct,
                "priority": predicted_priority,
                "priority_confidence": pri_conf_pct,
                "department": department,
                "confidence": cat_conf_pct,
                "error": None
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Prediction failed: {str(e)}",
                "category": "Unclassified",
                "category_confidence": 0.0,
                "priority": "Medium",
                "priority_confidence": 0.0,
                "department": "General Support Team",
                "confidence": 0.0
            }


# Singleton Predictor Instance
_predictor_instance = None


def get_predictor(models_dir: str = None) -> TicketPredictor:
    global _predictor_instance
    if _predictor_instance is None or not _predictor_instance.is_loaded:
        _predictor_instance = TicketPredictor(models_dir)
    return _predictor_instance


def predict_ticket(ticket_text: str, models_dir: str = None) -> dict:
    """Helper function to predict directly from raw text."""
    predictor = get_predictor(models_dir)
    return predictor.predict(ticket_text)
