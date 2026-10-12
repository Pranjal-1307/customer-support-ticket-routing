# Phase 13: Model Versioning & Artifact Traceability

## 1. Overview
M3 implements strict model lineage and artifact tracking to prevent unverified model changes and ensure complete auditability.

---

## 2. Model Version Registry (`models/model_version.json`)

```json
{
    "active_production_version": "v1.0",
    "candidate_version": "v2.0-candidate",
    "dataset_version": "2.0",
    "last_updated": "2026-09-26T19:25:00",
    "models": {
        "v1.0-production": {
            "artifact": "models/priority_classifier.pkl",
            "versioned_artifact": "models/priority_classifier_v1.pkl",
            "model_type": "Priority Classifier (M2 Baseline)",
            "algorithm": "Calibrated Linear SVM",
            "features": "TF-IDF (1,2)-grams (max_features=3000)",
            "training_samples": 3103,
            "synthetic_test_accuracy": 0.8953,
            "synthetic_test_macro_f1": 0.9068,
            "holdout_accuracy": 0.5167,
            "holdout_macro_f1": 0.5646,
            "status": "ACTIVE_PRODUCTION"
        },
        "v2.0-candidate": {
            "artifact": "models/m3_priority_classifier_candidate.pkl",
            "versioned_artifact": "models/priority_classifier_v2.pkl",
            "model_type": "Priority Classifier (M3 Candidate)",
            "algorithm": "Calibrated Linear SVM (Class-Weighted)",
            "features": "TF-IDF (1,2)-grams + 6 Priority Signals (Urgency, Severity, Financial Impact, Security Risk, Service Impact, Time Sensitivity)",
            "training_samples": 3103,
            "synthetic_test_accuracy": 0.9003,
            "synthetic_test_macro_f1": 0.9114,
            "holdout_accuracy": 0.5167,
            "realistic_scenario_accuracy": 0.6667,
            "status": "CANDIDATE_STAGED"
        }
    }
}
```

---

## 3. Production Artifact Integrity
- `models/priority_classifier_v1.pkl`: Exact frozen M2 baseline.
- `models/priority_classifier_v2.pkl`: Staged M3 candidate incorporating multi-dimensional priority signal features.
- `models/m3_priority_classifier_candidate.pkl`: Isolated evaluation artifact.
- `models/priority_classifier.pkl`: Active production model endpoint.
