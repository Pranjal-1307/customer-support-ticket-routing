"""
Milestone 2 — Automated Test Suite
===================================
Tests for data validation, incoming ticket pipeline schema, PII detection,
deduplication, dataset versioning, multilingual handling, synthetic generator diversity,
zero-leakage template group splitting, deterministic priority assignment, and application regression.

Usage:
    python -m pytest tests/ -v
"""

import os
import sys
import json
import pytest
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.predict import get_predictor, predict_ticket
from utils.route import get_department_for_category
from scripts.ingest_incoming import validate_incoming_record, check_pii
from scripts.prepare_dataset import normalize_text_pattern
from scripts.generate_data import assign_deterministic_priority, generate_tickets


# ═══════════════════════════════════════════════════════════════════════
# DATA VALIDATION & PII TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestDataValidationAndPII:
    def test_pii_email_detection(self):
        text = "Please send confirmation to test.user@company.org regarding my bill."
        pii = check_pii(text)
        assert len(pii) > 0
        assert any("Email" in p for p in pii)

    def test_pii_phone_detection(self):
        text = "Contact me at 555-123-4567 for urgent refund update."
        pii = check_pii(text)
        assert len(pii) > 0
        assert any("Phone" in p for p in pii)

    def test_pii_clean_text(self):
        text = "Why am I seeing a duplicate charge of 49 dollars on my account statement?"
        pii = check_pii(text)
        assert len(pii) == 0

    def test_validate_incoming_record_valid(self):
        row = pd.Series({
            "ticket_id": "TEST_INC_999",
            "ticket_text": "Why am I seeing two charges on my account for order #12345?",
            "category": "Billing",
            "priority": "High",
            "source": "external"
        })
        is_valid, reasons = validate_incoming_record(row, existing_ids=set(), existing_texts=set())
        assert is_valid is True
        assert len(reasons) == 0

    def test_validate_incoming_record_invalid_category(self):
        row = pd.Series({
            "ticket_id": "TEST_INC_998",
            "ticket_text": "Need help with my account password reset",
            "category": "InvalidCategoryName",
            "priority": "High",
            "source": "external"
        })
        is_valid, reasons = validate_incoming_record(row, existing_ids=set(), existing_texts=set())
        assert is_valid is False
        assert any("Invalid category" in r for r in reasons)

    def test_validate_incoming_record_short_text(self):
        row = pd.Series({
            "ticket_id": "TEST_INC_997",
            "ticket_text": "bad",
            "category": "Billing",
            "priority": "Low",
            "source": "external"
        })
        is_valid, reasons = validate_incoming_record(row, existing_ids=set(), existing_texts=set())
        assert is_valid is False
        assert any("too short" in r for r in reasons)

    def test_validate_incoming_record_duplicate_id(self):
        row = pd.Series({
            "ticket_id": "TICK-000001",
            "ticket_text": "Where is my package delivery?",
            "category": "Shipping",
            "priority": "Medium",
            "source": "external"
        })
        is_valid, reasons = validate_incoming_record(row, existing_ids={"TICK-000001"}, existing_texts=set())
        assert is_valid is False
        assert any("Duplicate ticket_id" in r for r in reasons)


# ═══════════════════════════════════════════════════════════════════════
# DATASET VERSIONING TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestDatasetVersioning:
    def test_dataset_version_metadata_exists(self):
        meta_path = os.path.join(BASE_DIR, "models", "dataset_version.json")
        assert os.path.exists(meta_path), "Missing dataset_version.json"

    def test_dataset_version_metadata_fields(self):
        meta_path = os.path.join(BASE_DIR, "models", "dataset_version.json")
        with open(meta_path, "r") as f:
            data = json.load(f)
        assert data.get("version") == "v2.0-M2"
        assert "timestamp" in data
        assert "source_counts" in data
        assert data.get("total_cleaned_records", 0) > 0
        assert "splits" in data
        assert data["splits"].get("train_size", 0) > 0

    def test_dataset_version_report_markdown_exists(self):
        report_path = os.path.join(BASE_DIR, "reports", "m2", "dataset_version.md")
        assert os.path.exists(report_path), "Missing dataset_version.md"


# ═══════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATION & DIVERSITY TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestSyntheticGeneratorDiversity:
    def test_generate_tickets_count_and_columns(self):
        df = generate_tickets(100)
        assert len(df) >= 90
        required_cols = ["ticket_id", "ticket_text", "category", "priority", "department", "source", "language"]
        for col in required_cols:
            assert col in df.columns

    def test_length_variation_in_generator(self):
        df = generate_tickets(200)
        words = df["ticket_text"].apply(lambda t: len(str(t).split()))
        assert words.min() < 10, "Generator missing short ticket examples"
        assert words.max() > 20, "Generator missing long ticket examples"

    def test_multilingual_tickets_generated(self):
        df = generate_tickets(400)
        languages = set(df["language"].unique())
        assert "hinglish" in languages or "hindi" in languages or "gujarati" in languages, "Generator missing code-mixed language examples"

    def test_deterministic_priority_assignment(self):
        text_critical = "System downtime during event caused outage"
        p_crit = assign_deterministic_priority(text_critical, "High", "")
        assert p_crit == "Critical"

        text_low = "Where can I find documentation for dark mode?"
        p_low = assign_deterministic_priority(text_low, "Medium", "")
        assert p_low == "Low"


# ═══════════════════════════════════════════════════════════════════════
# MULTILINGUAL PREDICTION TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestMultilingualPrediction:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.predictor = get_predictor()

    def test_hinglish_billing_prediction(self):
        res = self.predictor.predict("Mera bank account se double payment deduct ho gaya hai, please refund karo")
        assert res["status"] == "success"
        assert res["category"] in ["Billing", "Refund"]
        assert res["priority"] in ["High", "Critical", "Medium"]

    def test_hinglish_technical_support_prediction(self):
        res = self.predictor.predict("App launch hone par crash ho raha hai 500 server error ke saath")
        assert res["status"] == "success"
        assert res["category"] == "Technical Support"

    def test_gujarati_shipping_prediction(self):
        res = self.predictor.predict("Aa order ma shipping status change nathi thayo 5 din thi")
        assert res["status"] == "success"
        assert res["category"] == "Shipping"


# ═══════════════════════════════════════════════════════════════════════
# APPLICATION REGRESSION TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestApplicationRegression:
    def test_flask_app_import_and_client(self):
        from app.app import app
        client = app.test_client()
        response = client.get("/")
        assert response.status_code == 200

    def test_api_predict_endpoint(self):
        from app.app import app
        client = app.test_client()
        response = client.post("/api/predict", json={
            "ticket_text": "My credit card was billed twice for transaction #99201."
        })
        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["status"] == "success"
        assert "category" in json_data
        assert "priority" in json_data
        assert "department" in json_data
        assert "confidence" in json_data

    def test_api_submit_endpoint_and_db(self):
        from app.app import app
        client = app.test_client()
        response = client.post("/api/tickets/submit", json={
            "ticket_text": "Unable to log into my admin account due to expired 2FA code."
        })
        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["status"] == "success"
        assert "ticket_id" in json_data
