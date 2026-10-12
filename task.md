# Project Milestones & Task Progress

## M1 — Baseline Machine Learning & Data Pipeline [COMPLETED]
- [x] Initial dataset generation and preprocessing
- [x] TF-IDF feature extraction (1,2-grams)
- [x] Multi-class category classification models (Naive Bayes, Logistic Regression, Random Forest, Linear SVM)
- [x] SQLite database schema and ticket persistence
- [x] Flask REST API and customer/admin UI
- [x] Initial unit tests

## M2 — Confidence Calibration & Diagnostic Auditing [COMPLETED]
- [x] Preprocessing robustness and lemmatization pipeline
- [x] Probability calibration via Platt scaling / CalibratedClassifierCV
- [x] Category & Priority confidence separation
- [x] Diagnostic holdout audit revealing synthetic priority bias (51.67% holdout acc)
- [x] 87 unit and integration tests passing

## M3 — Real-World Priority Intelligence & Human Feedback [COMPLETED]
- [x] **Phase 0 — Protect M2:** Frozen commit `0ead4b19d5dccb5be6ebd24f968a3e8a1e6b32af`, created `reports/m3/m3_baseline.md`
- [x] **Phase 1 — Reproduce Priority Problem:** Measured failure modes on 10 canonical test cases with probability distributions
- [x] **Phase 2 — Data Forensics:** Audited lexical topic confounding and vocabulary gaps in `reports/m3/priority_data_forensics.md`
- [x] **Phase 3 — Language Bias Investigation:** Audited Hinglish vs English divergence in `reports/m3/language_analysis.md`
- [x] **Phase 4 — Real-World Priority Signals:** Created `utils/priority_signals.py` (Urgency, Severity, Financial Impact, Security Risk, Service Impact, Time Sensitivity)
- [x] **Phase 5 — Better Feature Pipeline:** Stacked TF-IDF + Priority Signals without modifying production artifacts
- [x] **Phase 6 — Model Comparison:** Benchmarked 6 architectures on identical splits in `reports/m3/model_comparison.md`
- [x] **Phase 7 — Realistic Priority Evaluation:** Evaluated on realistic scenarios (Accuracy improved from 41.7% to 66.7%) in `reports/m3/realistic_priority_evaluation.md`
- [x] **Phase 8 — Priority Explanations:** Integrated human-interpretable reasoning into prediction outputs
- [x] **Phase 9–11 — Human Feedback & Admin UI:** Added `priority_feedback` SQLite table, API endpoints (`POST`/`GET` `/api/feedback/priority`, `/validate`), and admin modal
- [x] **Phase 12 — Validation Gate:** Implemented `scripts/retrain_with_feedback.py` enforcing holdout quality thresholds before promotion
- [x] **Phase 13 — Model Versioning:** Structured `models/model_version.json` and explicit model artifacts (`v1.0`, `v2.0`)
- [x] **Phase 14 — Testing:** Added `tests/test_m3.py`; all 103 tests passing
- [x] **Phase 15 — Documentation:** Comprehensive reports created in `reports/m3/`
