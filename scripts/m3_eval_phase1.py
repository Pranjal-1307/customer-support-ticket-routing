"""
Phase 1: Reproduce M2 Priority Problem on Canonical Test Cases.
"""

import os
import sys
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.predict import get_predictor
from utils.preprocess import preprocess_text

def run_phase1_evaluation():
    p = get_predictor()
    test_cases = [
        "Where is my parcel?",
        "Where can I find my order tracking information?",
        "The parcel delivery is late",
        "My package is two days late",
        "My credit card was charged twice",
        "I want a refund. I was charged twice.",
        "My account was hacked",
        "I need to update my profile information.",
        "Can you tell me my order status?",
        "My package has been delayed for two weeks."
    ]

    results = []
    classes = list(p.priority_classifier.classes_)
    
    print(f"Priority Classes: {classes}")
    print("=" * 80)
    for idx, text in enumerate(test_cases, 1):
        res = p.predict(text)
        processed = preprocess_text(text)
        vec = p.vectorizer.transform([processed])
        pri_probs = p.priority_classifier.predict_proba(vec)[0]
        pri_dist = {cls: round(float(prob) * 100, 2) for cls, prob in zip(classes, pri_probs)}
        
        entry = {
            "index": idx,
            "ticket_text": text,
            "category": res["category"],
            "category_confidence": res["category_confidence"],
            "priority": res["priority"],
            "priority_confidence": res["priority_confidence"],
            "priority_distribution": pri_dist,
            "department": res["department"]
        }
        results.append(entry)
        
        print(f"Test Case {idx}: \"{text}\"")
        print(f"  Category: {res['category']} ({res['category_confidence']}%)")
        print(f"  Priority: {res['priority']} ({res['priority_confidence']}%)")
        print(f"  Distribution: {pri_dist}")
        print(f"  Department: {res['department']}")
        print("-" * 60)
        
    return results

if __name__ == "__main__":
    run_phase1_evaluation()
