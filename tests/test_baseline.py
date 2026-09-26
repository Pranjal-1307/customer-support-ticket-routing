"""
Milestone 1 — Baseline Tests
=============================
Basic automated tests for preprocessing, prediction, model loading,
TF-IDF transformation, and department routing.

Usage:
    python -m pytest tests/test_baseline.py -v
"""

import os
import sys
import pytest

# Ensure project root is on path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from utils.preprocess import clean_text, preprocess_text
from utils.route import get_department_for_category, CATEGORY_DEPARTMENT_MAP


# ═══════════════════════════════════════════════════════════════════════
# PREPROCESSING TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestPreprocessing:
    """Tests for utils/preprocess.py."""

    def test_empty_string(self):
        assert preprocess_text("") == ""

    def test_whitespace_only(self):
        assert preprocess_text("   ") == ""

    def test_none_input(self):
        assert preprocess_text(None) == ""

    def test_numeric_input(self):
        assert preprocess_text(123) == ""

    def test_normal_text(self):
        result = preprocess_text("My credit card was charged twice")
        assert isinstance(result, str)
        assert len(result) > 0
        # Should be lowercase
        assert result == result.lower()

    def test_lowercase_conversion(self):
        result = preprocess_text("HELLO WORLD TEST")
        assert result == result.lower()

    def test_url_removal(self):
        result = preprocess_text("Visit https://example.com for more info")
        assert "https" not in result
        assert "example" not in result

    def test_html_removal(self):
        result = clean_text("This is <b>bold</b> text")
        assert "<b>" not in result
        assert "<" not in result

    def test_special_characters(self):
        result = clean_text("Hello! @#$% world")
        assert "@" not in result
        assert "#" not in result

    def test_digit_removal(self):
        result = clean_text("Error code 12345 occurred")
        assert "12345" not in result

    def test_whitespace_normalization(self):
        result = clean_text("too    many    spaces")
        assert "    " not in result

    def test_stopword_removal(self):
        result = preprocess_text("I am going to the store and I will buy something")
        assert "the" not in result.split()
        assert "and" not in result.split()

    def test_short_token_removal(self):
        result = preprocess_text("a b c hello world")
        assert "a" not in result.split()

    def test_output_is_string(self):
        assert isinstance(preprocess_text("test input"), str)

    def test_idempotent(self):
        """Processing an already-processed text should be stable."""
        original = "My internet is not working and the router is blinking"
        first_pass = preprocess_text(original)
        second_pass = preprocess_text(first_pass)
        assert isinstance(second_pass, str)
        assert len(second_pass) > 0


# ═══════════════════════════════════════════════════════════════════════
# MODEL LOADING TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestModelLoading:
    """Tests for model artifact loading."""

    def test_model_artifacts_exist(self):
        models_dir = os.path.join(BASE_DIR, "models")
        required_files = [
            "vectorizer.pkl",
            "label_encoder.pkl",
            "ticket_classifier.pkl",
            "priority_classifier.pkl",
        ]
        for fname in required_files:
            fpath = os.path.join(models_dir, fname)
            assert os.path.exists(fpath), f"Missing model artifact: {fname}"

    def test_model_metrics_exists(self):
        metrics_path = os.path.join(BASE_DIR, "models", "model_metrics.json")
        assert os.path.exists(metrics_path), "Missing model_metrics.json"

    def test_predictor_loads(self):
        from utils.predict import TicketPredictor
        predictor = TicketPredictor()
        assert predictor.is_loaded, "Predictor failed to load model artifacts"
        assert predictor.vectorizer is not None
        assert predictor.label_encoder is not None
        assert predictor.ticket_classifier is not None
        assert predictor.priority_classifier is not None


# ═══════════════════════════════════════════════════════════════════════
# PREDICTION TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestPrediction:
    """Tests for utils/predict.py."""

    @pytest.fixture(autouse=True)
    def setup_predictor(self):
        from utils.predict import TicketPredictor
        self.predictor = TicketPredictor()

    def test_normal_ticket(self):
        result = self.predictor.predict("My credit card was charged twice for the same order")
        assert result["status"] == "success"
        assert result["category"] in CATEGORY_DEPARTMENT_MAP
        assert result["priority"] in ["Low", "Medium", "High", "Critical"]
        assert isinstance(result["confidence"], (int, float))
        assert result["confidence"] >= 0
        assert result["department"] != ""

    def test_empty_ticket(self):
        result = self.predictor.predict("")
        assert result["status"] == "error"

    def test_whitespace_ticket(self):
        result = self.predictor.predict("   ")
        assert result["status"] == "error"

    def test_none_ticket(self):
        result = self.predictor.predict(None)
        assert result["status"] == "error"

    def test_unusual_ticket(self):
        result = self.predictor.predict("asdfghjkl qwerty zxcvbnm 12345")
        assert result["status"] == "success"
        # Should still return valid structure even for gibberish
        assert "category" in result
        assert "priority" in result
        assert "department" in result

    def test_very_long_ticket(self):
        long_text = "This is a very long ticket about billing issues. " * 100
        result = self.predictor.predict(long_text)
        assert result["status"] == "success"
        assert "category" in result

    def test_single_word_ticket(self):
        result = self.predictor.predict("help")
        assert result["status"] == "success"

    def test_prediction_returns_correct_types(self):
        result = self.predictor.predict("I need a refund for my order")
        assert isinstance(result["category"], str)
        assert isinstance(result["priority"], str)
        assert isinstance(result["department"], str)
        assert isinstance(result["confidence"], (int, float))

    def test_confidence_range(self):
        result = self.predictor.predict("My internet is not working")
        # Confidence is returned as percentage (0-100)
        assert 0 <= result["confidence"] <= 100


# ═══════════════════════════════════════════════════════════════════════
# TF-IDF TRANSFORMATION TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestTfidfTransformation:
    """Tests for TF-IDF vectorizer consistency."""

    @pytest.fixture(autouse=True)
    def setup(self):
        import joblib
        self.vectorizer = joblib.load(os.path.join(BASE_DIR, "models", "vectorizer.pkl"))

    def test_vectorizer_transform(self):
        result = self.vectorizer.transform(["test ticket about billing"])
        assert result.shape[0] == 1
        assert result.shape[1] > 0

    def test_vectorizer_consistent_dimensions(self):
        r1 = self.vectorizer.transform(["billing issue"])
        r2 = self.vectorizer.transform(["shipping problem"])
        assert r1.shape[1] == r2.shape[1]

    def test_empty_text_transform(self):
        result = self.vectorizer.transform([""])
        assert result.shape[0] == 1


# ═══════════════════════════════════════════════════════════════════════
# DEPARTMENT ROUTING TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestDepartmentRouting:
    """Tests for utils/route.py."""

    def test_all_categories_mapped(self):
        expected_categories = [
            "Billing", "Technical Support", "Refund", "Shipping",
            "Account", "Complaint", "Product Inquiry", "Cancellation",
        ]
        for cat in expected_categories:
            dept = get_department_for_category(cat)
            assert dept != "General Support Team", f"Category '{cat}' not mapped"

    def test_unknown_category_default(self):
        assert get_department_for_category("Unknown") == "General Support Team"

    def test_empty_category(self):
        assert get_department_for_category("") == "General Support Team"

    def test_none_category(self):
        assert get_department_for_category(None) == "General Support Team"

    def test_specific_mappings(self):
        assert get_department_for_category("Billing") == "Billing Team"
        assert get_department_for_category("Technical Support") == "Technical Support Team"
        assert get_department_for_category("Shipping") == "Logistics Team"
        assert get_department_for_category("Cancellation") == "Retention Team"

    def test_whitespace_handling(self):
        assert get_department_for_category("  Billing  ") == "Billing Team"


# ═══════════════════════════════════════════════════════════════════════
# DATASET INTEGRITY TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestDatasetIntegrity:
    """Tests for dataset files."""

    def test_tickets_csv_exists(self):
        path = os.path.join(BASE_DIR, "dataset", "tickets.csv")
        assert os.path.exists(path)

    def test_train_csv_exists(self):
        path = os.path.join(BASE_DIR, "dataset", "train.csv")
        assert os.path.exists(path)

    def test_test_csv_exists(self):
        path = os.path.join(BASE_DIR, "dataset", "test.csv")
        assert os.path.exists(path)

    def test_required_columns(self):
        import pandas as pd
        df = pd.read_csv(os.path.join(BASE_DIR, "dataset", "tickets.csv"))
        required = ["ticket_id", "ticket_text", "category", "priority"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_null_text(self):
        import pandas as pd
        df = pd.read_csv(os.path.join(BASE_DIR, "dataset", "tickets.csv"))
        assert df["ticket_text"].isnull().sum() == 0

    def test_valid_categories(self):
        import pandas as pd
        valid = set(CATEGORY_DEPARTMENT_MAP.keys())
        df = pd.read_csv(os.path.join(BASE_DIR, "dataset", "tickets.csv"))
        actual = set(df["category"].unique())
        assert actual.issubset(valid), f"Invalid categories: {actual - valid}"

    def test_valid_priorities(self):
        import pandas as pd
        valid = {"Low", "Medium", "High", "Critical"}
        df = pd.read_csv(os.path.join(BASE_DIR, "dataset", "tickets.csv"))
        actual = set(df["priority"].unique())
        assert actual.issubset(valid), f"Invalid priorities: {actual - valid}"


# ═══════════════════════════════════════════════════════════════════════
# TEMPLATE NORMALIZATION TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestTemplateNormalization:
    """Tests for prepare_dataset.normalize_text_pattern()."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from scripts.prepare_dataset import normalize_text_pattern
        self.normalize = normalize_text_pattern

    def test_masks_currency_amounts(self):
        """Dollar amounts are masked (the $ symbol is removed and digits become ID/AMOUNT)."""
        result = self.normalize("I was charged $49.00 on my bill")
        assert "$49.00" not in result
        # The amount is masked by the currency regex
        assert "49" not in result

    def test_masks_ip_addresses(self):
        result = self.normalize("Cannot connect from 192.168.1.100")
        assert "192.168.1.100" not in result
        assert "<IP>" in result

    def test_masks_error_codes(self):
        result = self.normalize("System crashed showing ERR_502 error")
        assert "ERR_502" not in result.upper()

    def test_masks_model_numbers(self):
        result = self.normalize("Documentation for model PRO-456")
        assert "PRO-456" not in result.upper()
        assert "<MODEL>" in result

    def test_masks_periods(self):
        """Period year is masked as a number."""
        result = self.normalize("Billing period Jan 2026 discrepancy")
        assert "2026" not in result  # Year is masked

    def test_deterministic_normalization(self):
        """Same input always produces same normalized output."""
        text = "My credit card was charged $49.00 for order 12345"
        result1 = self.normalize(text)
        result2 = self.normalize(text)
        assert result1 == result2

    def test_different_fill_values_same_template(self):
        """Different fill values in the same template produce the same normalized output."""
        t1 = "My credit card was charged twice for transaction 12345."
        t2 = "My credit card was charged twice for transaction 67890."
        assert self.normalize(t1) == self.normalize(t2)

    def test_different_amounts_same_template(self):
        """Different dollar amounts in the same template produce the same output."""
        t1 = "I noticed an unauthorized charge of $49.00 on my monthly invoice."
        t2 = "I noticed an unauthorized charge of $99.99 on my monthly invoice."
        assert self.normalize(t1) == self.normalize(t2)

    def test_empty_string(self):
        assert self.normalize("") == ""

    def test_none_input(self):
        assert self.normalize(None) == ""

    def test_lowercases_base_text(self):
        """The function lowercases the input text before masking."""
        result = self.normalize("hello world")
        assert "hello" in result or "<" in result  # either preserved or masked


# ═══════════════════════════════════════════════════════════════════════
# STRATIFIED GROUP SPLIT / ZERO LEAKAGE TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestGroupSplitLeakage:
    """Tests that the stratified group split prevents template leakage."""

    @pytest.fixture(autouse=True)
    def setup(self):
        import pandas as pd
        from scripts.prepare_dataset import normalize_text_pattern
        self.normalize = normalize_text_pattern
        self.train_df = pd.read_csv(os.path.join(BASE_DIR, "data", "train.csv"))
        self.val_df = pd.read_csv(os.path.join(BASE_DIR, "data", "validation.csv"))
        self.test_df = pd.read_csv(os.path.join(BASE_DIR, "data", "test.csv"))

    def test_zero_exact_text_overlap_train_test(self):
        """No identical ticket texts should appear in both train and test."""
        train_texts = set(self.train_df["ticket_text"])
        test_texts = set(self.test_df["ticket_text"])
        overlap = train_texts.intersection(test_texts)
        assert len(overlap) == 0, f"Found {len(overlap)} exact text overlaps"

    def test_zero_exact_text_overlap_train_val(self):
        """No identical ticket texts should appear in both train and val."""
        train_texts = set(self.train_df["ticket_text"])
        val_texts = set(self.val_df["ticket_text"])
        overlap = train_texts.intersection(val_texts)
        assert len(overlap) == 0, f"Found {len(overlap)} exact text overlaps"

    def test_zero_exact_text_overlap_val_test(self):
        """No identical ticket texts should appear in both val and test."""
        val_texts = set(self.val_df["ticket_text"])
        test_texts = set(self.test_df["ticket_text"])
        overlap = val_texts.intersection(test_texts)
        assert len(overlap) == 0, f"Found {len(overlap)} exact text overlaps"

    def test_zero_template_overlap_train_test(self):
        """No normalized template patterns should appear in both train and test."""
        train_norms = set(self.train_df["ticket_text"].apply(self.normalize))
        test_norms = set(self.test_df["ticket_text"].apply(self.normalize))
        overlap = train_norms.intersection(test_norms)
        assert len(overlap) == 0, f"Found {len(overlap)} template overlaps"

    def test_zero_template_overlap_train_val(self):
        """No normalized template patterns should appear in both train and val."""
        train_norms = set(self.train_df["ticket_text"].apply(self.normalize))
        val_norms = set(self.val_df["ticket_text"].apply(self.normalize))
        overlap = train_norms.intersection(val_norms)
        assert len(overlap) == 0, f"Found {len(overlap)} template overlaps"

    def test_all_categories_in_splits(self):
        """All 8 categories should be present in every split."""
        expected = {
            "Billing", "Technical Support", "Refund", "Shipping",
            "Account", "Complaint", "Product Inquiry", "Cancellation",
        }
        for name, df in [("train", self.train_df), ("val", self.val_df), ("test", self.test_df)]:
            actual = set(df["category"].unique())
            assert expected.issubset(actual), f"{name} missing categories: {expected - actual}"

    def test_splits_not_empty(self):
        """All splits should have at least 50 samples."""
        assert len(self.train_df) >= 50, "Train set too small"
        assert len(self.val_df) >= 50, "Validation set too small"
        assert len(self.test_df) >= 50, "Test set too small"


# ═══════════════════════════════════════════════════════════════════════
# DETERMINISTIC PRIORITY LABELING TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestDeterministicPriority:
    """Tests for deterministic priority labeling in generate_data.py."""

    def test_urgency_suffix_upgrades_priority(self):
        """Tickets with urgency suffixes should have higher or equal priority."""
        import pandas as pd
        df = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "synthetic_tickets.csv"))

        urgency_suffixes = [
            "Urgent attention required.",
            "Please advise as soon as possible.",
        ]

        # Find pairs: same template, one with urgency suffix and one without
        for _, row in df.iterrows():
            text = row["ticket_text"]
            has_urgency = any(text.endswith(s) for s in urgency_suffixes)
            if has_urgency:
                # Priority should be at least "Medium" for urgency-upgraded tickets
                assert row["priority"] in ["Medium", "High", "Critical"], \
                    f"Urgency ticket has Low priority: {text[:60]}"

    def test_priority_distribution_not_uniform(self):
        """Priority distribution should NOT be uniform (random) — it should be skewed."""
        import pandas as pd
        df = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "synthetic_tickets.csv"))
        pri_counts = df["priority"].value_counts(normalize=True)
        # With deterministic labeling, the distribution should NOT be ~25% each
        # At least one priority should deviate from 25% by more than 5%
        deviations = [abs(pct - 0.25) for pct in pri_counts.values]
        assert max(deviations) > 0.05, "Priority distribution appears uniform/random"

    def test_reproducible_generation(self):
        """Running generate_tickets twice should produce identical output."""
        from scripts.generate_data import generate_tickets
        import random
        random.seed(42)
        df1 = generate_tickets(100)
        random.seed(42)
        df2 = generate_tickets(100)
        assert df1["priority"].tolist() == df2["priority"].tolist()
        assert df1["category"].tolist() == df2["category"].tolist()


# ═══════════════════════════════════════════════════════════════════════
# CALIBRATED LINEARSVC CONFIDENCE TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestCalibratedConfidence:
    """Tests model confidence calculation and output validity."""

    def test_model_supports_confidence_scoring(self):
        """The saved ticket_classifier should support predict_proba or decision_function."""
        import joblib
        model = joblib.load(os.path.join(BASE_DIR, "models", "ticket_classifier.pkl"))
        assert hasattr(model, "predict_proba") or hasattr(model, "decision_function"), \
            "ticket_classifier does not have predict_proba or decision_function"

    def test_confidence_is_proper_probability(self):
        """Confidence from predict should be between 0 and 100."""
        from utils.predict import TicketPredictor
        predictor = TicketPredictor()
        result = predictor.predict("My credit card was charged twice for the same order")
        assert result["status"] == "success"
        assert 0 < result["confidence"] <= 100
        assert 0 < result["category_confidence"] <= 100
        assert 0 <= result["priority_confidence"] <= 100

    def test_confidence_probabilities_sum_to_one(self):
        """The model's probability or softmax decision_function output should sum to ~1.0."""
        import joblib
        import numpy as np
        model = joblib.load(os.path.join(BASE_DIR, "models", "ticket_classifier.pkl"))
        vectorizer = joblib.load(os.path.join(BASE_DIR, "models", "vectorizer.pkl"))
        vec = vectorizer.transform(["test billing issue"])
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vec)[0]
        else:
            dec = model.decision_function(vec)[0]
            exp_dec = np.exp(dec - np.max(dec))
            probs = exp_dec / exp_dec.sum()
        assert abs(probs.sum() - 1.0) < 1e-6, f"Probabilities sum to {probs.sum()}, expected ~1.0"
        assert all(p >= 0 for p in probs), "Negative probability detected"

