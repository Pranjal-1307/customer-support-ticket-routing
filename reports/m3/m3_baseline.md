# M3 Baseline Protection & Initial State Report

## 1. Git Repository State
- **Branch:** `main`
- **Commit Hash:** `0ead4b19d5dccb5be6ebd24f968a3e8a1e6b32af`
- **Commit Message:** `Push M2`
- **Working Tree Status:** Clean (no uncommitted changes)
- **Baseline Preserved:** M2 model artifacts, datasets, tests, and API routes are frozen as baseline.

---

## 2. Production Model Artifacts & Architecture (M2 Baseline)
The production system currently uses the following artifacts located in `models/`:
- `vectorizer.pkl` (TF-IDF Vectorizer with ngram_range=(1,2), max_features=3000)
- `label_encoder.pkl` (Category LabelEncoder: Billing, Technical, Account, Shipping, General)
- `ticket_classifier.pkl` (Calibrated Linear SVM for category classification)
- `priority_classifier.pkl` (Calibrated Linear SVM for priority prediction: Low, Medium, High, Critical)
- `model_metrics.json` (Full M2 performance metrics across models)
- `dataset_version.json` (Dataset versioning metadata v2.0)

### Current Prediction Pipeline
```
Ticket Text 
   ↓ Preprocessing (lowercase, regex tokenize, lemmatize)
TF-IDF Vectorizer (1-2 ngrams)
   ↓
Category Classifier (Calibrated Linear SVM) → Category + Confidence
   ↓
Priority Classifier (Calibrated Linear SVM) → Priority + Confidence
   ↓
Department Routing (Deterministic rule based on predicted category)
   ↓
Database (SQLite tickets.db) / Web UI / REST API
```

---

## 3. Test Suite Verification
- **Total Tests Run:** 87 tests (`tests/test_baseline.py`, `tests/test_m2.py`)
- **Status:** All 87 passed in 6.49s.
- **Coverage:** Preprocessing, TF-IDF vectorization, Model loading & inference, Calibrated confidence scores, Category and Priority classification, Department routing, API endpoints (`/api/predict`, `/api/tickets/submit`, `/api/admin/stats`), Data leakage & synthetic bias checks.

---

## 4. M2 Baseline Performance Summary

| Metric | Category Model (Calibrated Linear SVM) | Priority Model (Calibrated Linear SVM) |
| :--- | :--- | :--- |
| **Synthetic Test Accuracy** | 99.34% | 89.53% |
| **Synthetic Test Macro F1** | 0.9933 | 0.9068 |
| **Synthetic Test Weighted F1** | 0.9934 | 0.8940 |
| **Realistic Holdout Accuracy** | 91.67% | **51.67%** |
| **Realistic Holdout Macro F1** | 0.9177 | **0.5646** |

### Key Diagnostic Observation
While synthetic priority test scores appear high (~89.5%), priority accuracy plummets to **51.67%** on real-world realistic holdout queries. This indicates that M2 priority prediction is heavily reliant on lexical topic tokens rather than true urgency/severity signals.

---

## 5. Next Steps for M3
1. **Phase 1:** Reproduce M2 priority failure modes on 10 canonical test cases with full probability distributions.
2. **Phase 2 & 3:** Perform comprehensive data forensics and language bias investigation.
3. **Phase 4:** Design interpretable multi-dimensional priority signal features (Urgency, Severity, Financial Impact, Security Risk, Service Impact, Time Sensitivity).
4. **Phase 5 & 6:** Build feature pipeline combining TF-IDF + Priority Signals, train and benchmark candidate models.
5. **Phase 7 & 8:** Evaluate on realistic priority dataset with interpretable reasoning outputs.
6. **Phase 9–13:** Implement Human Feedback correction workflow, validation gates, and formal model versioning without corrupting the M2 production baseline.
