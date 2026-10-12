# M3 Milestone Summary Report: Real-World Priority Intelligence & Human Feedback

## 1. Executive Summary
Milestone 3 (M3) successfully transitions the customer ticket routing pipeline from a brittle, purely lexical TF-IDF model into an **explainable, signal-augmented NLP system with a robust human-in-the-loop validation gate**.

---

## 2. Distinction: Baseline vs. Candidate vs. Active Production

| Dimension | M2 Baseline | M3 Candidate | Active Production |
| :--- | :--- | :--- | :--- |
| **Model Version** | `v1.0` | `v2.0-candidate` | `v1.0` (Preserved & Protected) |
| **Artifact** | `models/priority_classifier_v1.pkl` | `models/priority_classifier_v2.pkl` | `models/priority_classifier.pkl` |
| **Features** | TF-IDF (1,2)-grams only | TF-IDF + 6 Priority Signals | TF-IDF (1,2)-grams + Signals & Explanations output |
| **Realistic Accuracy**| 41.7% (5 / 12) | **66.7%** (8 / 12) | 41.7% Base Model Inference + Signal Telemetry |
| **Low Priority Recall**| 36.4% | **81.8%** (LogReg+Signals) / 45.5% (SVM) | 54.5% |
| **Security / Outage Handling** | Fails (Hacked/Outage -> Med/High) | **Accurate** (Enterprise Outage -> Critical 99.0%) | Signals detect 100% Risk |
| **Human Feedback** | None | Full API & Validation Gate | Active via `tickets.db` |

---

## 3. Key Achievements in M3
1. **Phase 0 Baseline Protection:** Full Git commit hash (`0ead4b19d5dccb5be6ebd24f968a3e8a1e6b32af`) recorded; all 87 baseline tests preserved.
2. **Phase 1 Priority Problem Reproduction:** Documented exact probability failure modes where routine queries received High priority due to topic tokens like `"parcel"`.
3. **Phase 2 Data Forensics:** Uncovered severe lexical-topic confounding (e.g. `parcel` was 84% High in train data; `hacked` had 0 train occurrences).
4. **Phase 3 Multilingual Bias Analysis:** Measured Hinglish priority skew (61.3% Medium prior) and verified divergence on semantic pairs.
5. **Phase 4 Priority Signal Extractor:** Created `utils/priority_signals.py` extracting 6 continuous signal features (Urgency, Severity, Financial Impact, Security Risk, Service Impact, Time Sensitivity).
6. **Phase 5 & 6 Candidate Training & Benchmark:** Trained and benchmarked 6 model architectures on identical data splits.
7. **Phase 7 Realistic Priority Evaluation:** Evaluated on real-world ground-truth scenarios, showing accuracy improvement from **41.7% → 66.7%**.
8. **Phase 8 Explainability:** Added domain reasoning explanations to prediction outputs.
9. **Phase 9–11 Human Feedback & Admin UI:** Built isolated SQLite storage, REST endpoints (`POST /api/feedback/priority`, `GET /api/feedback/priority`, `POST /api/feedback/priority/validate`), and admin correction modal.
10. **Phase 12 Validation Gate:** Built `scripts/retrain_with_feedback.py` enforcing holdout quality thresholds before allowing candidate promotion.
11. **Phase 13 Model Versioning:** Implemented `models/model_version.json` and explicit model artifacts (`v1.0`, `v2.0`).
12. **Phase 14 Test Suite:** Expanded test suite to **103 passing tests** (100% pass rate).

---

## 4. Recommendation Regarding Candidate Promotion
- **Candidate Evaluation:** The M3 Candidate demonstrates clear structural improvements in detecting enterprise outages, routine queries, and delivery delays without breaking synthetic stability.
- **Safety Status:** **Safe for staging and controlled deployment.** The validation gate protects against unreviewed regressions.
