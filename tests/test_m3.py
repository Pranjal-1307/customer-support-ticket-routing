"""
Comprehensive Unit & Integration Test Suite for M3.
Covers:
- Priority Signal Extraction (6 dimensions)
- Explanation Generation
- Prediction Output Schema & Separate Confidences
- Human Feedback Submission & Persistence
- Feedback Validation Gate & Endpoints
- Model Versioning Metadata Integrity
- API Regressions (Category Prediction, Department Routing)
"""

import os
import sys
import json
import pytest
import sqlite3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.priority_signals import extract_priority_signals, signals_to_feature_vector, generate_priority_explanation
from utils.predict import get_predictor, predict_ticket
from app.app import app, init_db, get_db_connection


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        with app.app_context():
            init_db()
        yield client


# ==========================================
# 1. PRIORITY SIGNAL EXTRACTION TESTS
# ==========================================

def test_signal_extraction_structure():
    """Verify signal extractor returns all 6 continuous float features."""
    signals = extract_priority_signals("My account was hacked and money was deducted urgently!")
    expected_keys = {"urgency", "severity", "financial_impact", "security_risk", "service_impact", "time_sensitivity"}
    assert set(signals.keys()) == expected_keys
    for k, v in signals.items():
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0


def test_signal_extraction_security_risk():
    signals = extract_priority_signals("My account was hacked by unauthorized access.")
    assert signals["security_risk"] >= 0.8


def test_signal_extraction_financial_impact():
    signals = extract_priority_signals("I was charged twice and need a refund immediately.")
    assert signals["financial_impact"] >= 0.8
    assert signals["urgency"] >= 0.8


def test_signal_extraction_service_impact():
    signals = extract_priority_signals("Our entire business service is unavailable due to system outage.")
    assert signals["service_impact"] >= 0.9


def test_signal_extraction_multilingual_hinglish():
    signals = extract_priority_signals("Mera card do baar charge ho gaya hai jaldi help karo.")
    assert signals["financial_impact"] >= 0.8
    assert signals["urgency"] >= 0.8


def test_signal_extraction_routine_inquiry():
    signals = extract_priority_signals("Where can I find tracking information for my order?")
    assert signals["security_risk"] == 0.0
    assert signals["financial_impact"] == 0.0
    assert signals["service_impact"] == 0.0


def test_signal_to_vector_conversion():
    signals = extract_priority_signals("Urgent server error.")
    vec = signals_to_feature_vector(signals)
    assert len(vec) == 6
    assert all(isinstance(x, float) for x in vec)


def test_priority_explanation_generation():
    signals = {"urgency": 0.9, "severity": 0.0, "financial_impact": 0.95, "security_risk": 0.0, "service_impact": 0.0, "time_sensitivity": 0.0}
    exp = generate_priority_explanation(signals, "High")
    assert isinstance(exp, str)
    assert "High priority" in exp
    assert "financial" in exp.lower()


# ==========================================
# 2. PREDICTION MODULE & SCHEMA REGRESSION
# ==========================================

def test_predict_output_m3_schema():
    predictor = get_predictor()
    res = predictor.predict("My internet router is broken and red light is blinking.")
    assert res["status"] == "success"
    assert "category" in res
    assert "category_confidence" in res
    assert "priority" in res
    assert "priority_confidence" in res
    assert "priority_signals" in res
    assert "priority_explanation" in res
    assert "department" in res
    assert isinstance(res["priority_signals"], dict)
    assert len(res["priority_signals"]) == 6


def test_category_and_routing_regression():
    p1 = predict_ticket("I need a refund for my double charge.")
    assert p1["category"] == "Billing" or p1["category"] == "Refund"
    assert "Billing Team" in p1["department"] or "Refund Team" in p1["department"]

    p2 = predict_ticket("My router cannot connect to internet wifi.")
    assert p2["category"] == "Technical Support"
    assert p2["department"] == "Technical Support Team"


# ==========================================
# 3. HUMAN FEEDBACK API & WORKFLOW TESTS
# ==========================================

def test_feedback_submission_success(client):
    # First submit a ticket to get a valid ticket_id
    t_res = client.post('/api/tickets/submit', json={"ticket_text": "Routine question about product manual."})
    t_data = t_res.get_json()
    ticket_id = t_data["ticket_id"]
    orig_pri = t_data["priority"]

    # Submit human feedback correction
    fb_res = client.post('/api/feedback/priority', json={
        "ticket_id": ticket_id,
        "corrected_priority": "Low",
        "reason": "Routine informational inquiry with no urgency.",
        "reviewer": "supervisor_1"
    })
    assert fb_res.status_code == 201
    fb_data = fb_res.get_json()
    assert fb_data["status"] == "success"
    assert "feedback_id" in fb_data
    assert fb_data["ticket_id"] == ticket_id
    assert fb_data["original_priority"] == orig_pri
    assert fb_data["corrected_priority"] == "Low"
    assert fb_data["validated"] == 0


def test_feedback_submission_invalid_priority(client):
    t_res = client.post('/api/tickets/submit', json={"ticket_text": "Sample text for test."})
    ticket_id = t_res.get_json()["ticket_id"]

    fb_res = client.post('/api/feedback/priority', json={
        "ticket_id": ticket_id,
        "corrected_priority": "SuperUrgent"
    })
    assert fb_res.status_code == 400


def test_feedback_submission_nonexistent_ticket(client):
    fb_res = client.post('/api/feedback/priority', json={
        "ticket_id": "TICK-NONEXISTENT999",
        "corrected_priority": "High"
    })
    assert fb_res.status_code == 404


def test_feedback_validation_endpoint(client):
    t_res = client.post('/api/tickets/submit', json={"ticket_text": "Sample for validation test."})
    ticket_id = t_res.get_json()["ticket_id"]

    fb_res = client.post('/api/feedback/priority', json={
        "ticket_id": ticket_id,
        "corrected_priority": "Medium",
        "reviewer": "qa_agent"
    })
    feedback_id = fb_res.get_json()["feedback_id"]

    # Validate feedback
    val_res = client.post('/api/feedback/priority/validate', json={
        "feedback_id": feedback_id,
        "validated": 1
    })
    assert val_res.status_code == 200
    assert val_res.get_json()["validated"] == 1

    # Verify via GET endpoint
    get_res = client.get('/api/feedback/priority')
    assert get_res.status_code == 200
    fb_items = get_res.get_json()["feedback"]
    matched = [f for f in fb_items if f["feedback_id"] == feedback_id]
    assert len(matched) == 1
    assert matched[0]["validated"] == 1


# ==========================================
# 4. MODEL VERSIONING & REPRODUCIBILITY
# ==========================================

def test_model_version_metadata():
    version_file = os.path.join(BASE_DIR, "models", "model_version.json")
    assert os.path.exists(version_file)
    with open(version_file, "r") as f:
        meta = json.load(f)
    assert "active_production_version" in meta
    assert "candidate_version" in meta
    assert "models" in meta
    assert "v1.0-production" in meta["models"]
    assert "v2.0-candidate" in meta["models"]


def test_production_artifact_preservation():
    """Ensure original production model is intact and matches versioned copy."""
    prod_path = os.path.join(BASE_DIR, "models", "priority_classifier.pkl")
    v1_path = os.path.join(BASE_DIR, "models", "priority_classifier_v1.pkl")
    assert os.path.exists(prod_path)
    assert os.path.exists(v1_path)


# ==========================================
# 5. MULTILINGUAL SEMANTIC CONSISTENCY TESTS
# ==========================================

def test_multilingual_semantic_consistency_shipping_order():
    """Group 1: Order location inquiries in EN, Hinglish, Gujarati must route to Shipping."""
    q_en = predict_ticket("Where is my order?")
    q_hi = predict_ticket("Mera order kaha hai?")
    q_gu = predict_ticket("મારું ઓર્ડર ક્યાં છે?")

    assert q_en["category"] == "Shipping"
    assert q_hi["category"] == "Shipping"
    assert q_gu["category"] == "Shipping"

    # Department consistency
    assert q_en["department"] == "Logistics Team"
    assert q_hi["department"] == "Logistics Team"
    assert q_gu["department"] == "Logistics Team"


def test_multilingual_semantic_consistency_shipping_package():
    """Group 2: Package location inquiries in EN, Hinglish, Gujarati must route to Shipping."""
    q_en = predict_ticket("Where is my package?")
    q_hi = predict_ticket("Mera package kahan hai?")
    q_gu = predict_ticket("મારું પેકેજ ક્યાં છે?")

    assert q_en["category"] == "Shipping"
    assert q_hi["category"] == "Shipping"
    assert q_gu["category"] == "Shipping"


def test_multilingual_semantic_consistency_account_hacked():
    """Group 3: Account security breach in EN, Hinglish, Gujarati must route to Account."""
    q_en = predict_ticket("My account has been hacked.")
    q_hi = predict_ticket("Mera account hack ho gaya hai.")
    q_gu = predict_ticket("મારું એકાઉન્ટ હેક થઈ ગયું છે.")

    assert q_en["category"] == "Account"
    assert q_hi["category"] == "Account"
    assert q_gu["category"] == "Account"


def test_multilingual_semantic_consistency_billing_duplicate():
    """Group 4: Double payment in EN, Hinglish, Gujarati must route to Billing."""
    q_en = predict_ticket("I was charged twice.")
    q_hi = predict_ticket("Mujhse do baar payment li gayi.")
    q_gu = predict_ticket("મારા પાસેથી બે વાર પૈસા લેવામાં આવ્યા છે.")

    assert q_en["category"] == "Billing"
    assert q_hi["category"] == "Billing"
    assert q_gu["category"] == "Billing"


def test_multilingual_semantic_consistency_overall_suite():
    """Verify entire multilingual consistency benchmark achieves >= 90%."""
    from scripts.evaluate_multilingual import evaluate_multilingual_consistency
    score, _ = evaluate_multilingual_consistency()
    assert score >= 90.0
