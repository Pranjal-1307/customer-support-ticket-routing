"""
Phase 7: Realistic Priority Evaluation on Real-World Ground-Truth Scenarios.
"""

import os
import sys
import joblib
import pandas as pd
from scipy.sparse import hstack

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.preprocess import preprocess_text
from utils.predict import get_predictor
from utils.priority_signals import extract_priority_signals, signals_to_feature_vector, generate_priority_explanation

REALISTIC_SCENARIOS = [
    # LOW PRIORITY
    {
        "ticket_text": "Where is my order?",
        "expected_priority": "Low",
        "reasoning": "Routine informational status inquiry; no operational defect, delay, or financial distress."
    },
    {
        "ticket_text": "Can you tell me my order status?",
        "expected_priority": "Low",
        "reasoning": "Routine product status check without urgency or financial loss."
    },
    {
        "ticket_text": "Where can I find tracking information?",
        "expected_priority": "Low",
        "reasoning": "General navigational guidance regarding shipment tracking."
    },
    # MEDIUM PRIORITY
    {
        "ticket_text": "My package is delayed by two days.",
        "expected_priority": "Medium",
        "reasoning": "Minor shipment delivery delay requiring routine tracking investigation."
    },
    {
        "ticket_text": "I have been waiting for my refund.",
        "expected_priority": "Medium",
        "reasoning": "Standard refund inquiry with moderate customer expectation."
    },
    {
        "ticket_text": "I cannot update my billing address.",
        "expected_priority": "Medium",
        "reasoning": "Non-blocking profile configuration difficulty."
    },
    # HIGH PRIORITY
    {
        "ticket_text": "I was charged twice for the same order.",
        "expected_priority": "High",
        "reasoning": "Direct financial impact and duplicate billing needing rapid resolution."
    },
    {
        "ticket_text": "I cannot access my account and I need it today.",
        "expected_priority": "High",
        "reasoning": "Account lockout with an explicit strict deadline (same day urgency)."
    },
    {
        "ticket_text": "My payment failed during an important purchase.",
        "expected_priority": "High",
        "reasoning": "Checkout failure impacting a time-critical customer transaction."
    },
    # CRITICAL PRIORITY
    {
        "ticket_text": "My account has been hacked.",
        "expected_priority": "Critical",
        "reasoning": "Severe security compromise, active account hijacking, data protection risk."
    },
    {
        "ticket_text": "There are unauthorized transactions on my account.",
        "expected_priority": "Critical",
        "reasoning": "Active financial fraud, identity compromise requiring immediate account freezing."
    },
    {
        "ticket_text": "Our entire business service is unavailable.",
        "expected_priority": "Critical",
        "reasoning": "Total enterprise outage, critical business disruption affecting all operations."
    }
]

def run_realistic_eval():
    predictor = get_predictor()
    candidate_path = os.path.join(BASE_DIR, "models", "m3_priority_classifier_candidate.pkl")
    candidate_model = joblib.load(candidate_path) if os.path.exists(candidate_path) else None

    results = []
    print("=" * 80)
    print("REALISTIC PRIORITY BENCHMARK (Ground Truth vs M2 Baseline vs M3 Candidate)")
    print("=" * 80)

    for item in REALISTIC_SCENARIOS:
        text = item["ticket_text"]
        exp = item["expected_priority"]
        reason = item["reasoning"]

        # Predict with M2 Baseline
        m2_res = predictor.predict(text)
        m2_pred_pri = m2_res["priority"]
        m2_conf = m2_res["priority_confidence"]

        # Predict with M3 Candidate
        processed = preprocess_text(text) or "ticket"
        vec_tfidf = predictor.vectorizer.transform([processed])
        sig = extract_priority_signals(text)
        sig_vec = signals_to_feature_vector(sig)
        import numpy as np
        from scipy.sparse import csr_matrix
        sig_mat = csr_matrix(np.array([sig_vec], dtype=np.float32))
        comb = hstack([vec_tfidf, sig_mat]).tocsr()
        
        cand_pred_pri = candidate_model.predict(comb)[0] if candidate_model else "N/A"
        cand_probs = candidate_model.predict_proba(comb)[0] if candidate_model else [0.0]
        cand_conf = round(float(np.max(cand_probs)) * 100, 2) if candidate_model else 0.0
        cand_explanation = generate_priority_explanation(sig, cand_pred_pri)

        entry = {
            "ticket_text": text,
            "expected_priority": exp,
            "reasoning": reason,
            "m2_priority": m2_pred_pri,
            "m2_confidence": m2_conf,
            "m2_match": (m2_pred_pri == exp),
            "m3_candidate_priority": cand_pred_pri,
            "m3_candidate_confidence": cand_conf,
            "m3_candidate_match": (cand_pred_pri == exp),
            "signals": sig,
            "explanation": cand_explanation
        }
        results.append(entry)

        print(f"\nTicket: \"{text}\"")
        print(f"  Expected: {exp}")
        print(f"  Domain Reasoning: {reason}")
        print(f"  M2 Baseline Pred: {m2_pred_pri} ({m2_conf}%) -> {'[MATCH]' if m2_pred_pri == exp else '[FAIL]'}")
        print(f"  M3 Candidate Pred: {cand_pred_pri} ({cand_conf}%) -> {'[MATCH]' if cand_pred_pri == exp else '[FAIL]'}")
        print(f"  Extracted Signals: {sig}")
        print(f"  Generated Explanation: {cand_explanation}")

    m2_acc = sum(1 for r in results if r["m2_match"]) / len(results) * 100
    m3_acc = sum(1 for r in results if r["m3_candidate_match"]) / len(results) * 100
    print("\n" + "=" * 80)
    print(f"OVERALL REALISTIC SCENARIO ACCURACY:")
    print(f"  M2 Baseline:    {m2_acc:.1f}% ({sum(1 for r in results if r['m2_match'])}/{len(results)})")
    print(f"  M3 Candidate:   {m3_acc:.1f}% ({sum(1 for r in results if r['m3_candidate_match'])}/{len(results)})")
    print("=" * 80)

    return results

if __name__ == "__main__":
    run_realistic_eval()
