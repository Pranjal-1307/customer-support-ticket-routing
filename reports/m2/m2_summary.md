# Milestone 2 — Executive Summary Report

**Completed Date:** `2026-08-20T08:46:14.677662`  
**Dataset Version:** `v2.0-M2`  
**Milestone Status:** **PASS**  

---

## 1. Overview & Objective

Milestone 2 focused on enhancing dataset realism and language diversity while implementing a safe, validated incoming ticket pipeline for manual and external support tickets.

Key principles enforced:
- **No data fabrication or leakage:** Synthetic data remains clearly labeled (`source="synthetic"`). Unverified tickets are isolated in `data/incoming/` and never automatically added to training data.
- **Deterministic Priority Labeling:** Priority assignment follows domain impact rules (urgency, severity, financial/security impact, SLA sensitivity) across varied sentence structures.
- **Reproducible Pipeline:** Maintained 0% data leakage between train, validation, and test splits using canonical template pattern group splitting.

---

## 2. Key Dataset & Model Metrics (M1 vs M2)

| Metric / Dimension | M1 Baseline | M2 Realistic Pipeline | Delta / Status |
|---|---|---|---|
| **Total Raw Combined Records** | 1,521 | 5761 | +4,240 records |
| **Clean Processed Records** | 1,521 | 4341 | +2,820 records |
| **Unique Ticket Texts** | 1,521 | 4341 | +2,820 records |
| **Exact Train/Test Overlap** | 0 | 0 | **PASSED (0%)** |
| **Template Pattern Overlap** | Documented | 0 | **PASSED (0%)** |
| **Best Category Model** | Random Forest | Naive Bayes | Weighted F1: 0.9934 |
| **Best Priority Model** | Logistic Regression | Calibrated Linear SVM | Weighted F1: 0.9307 |
| **Out-of-Distribution Holdout Acc** | N/A | Category: 93.3% / Priority: 60.0% | Tested on 60 realistic tickets |

---

## 3. Dataset Source & Language Breakdown

### Source Breakdown
- **Synthetic Tickets:** 4180 (96.3%)
- **External Tickets:** 77 (1.8%)
- **Manual Tickets:** 84 (1.9%)

### Multilingual Breakdown
- **English:** 2,644 records (60.9%)
- **Hinglish / Hindi Code-Mixed:** 1,487 records (34.3%)
- **Gujarati Code-Mixed:** 210 records (4.8%)

---

## 4. Pipeline Safety & Validation Summary

- **Incoming Pipeline:** Incoming CSV files land in `data/incoming/` and are processed via `scripts/ingest_incoming.py`.
- **Validation Checks:** Missing fields, text length bounds (5–2000 chars), PII regex detection (emails, phone numbers, credit cards, SSNs), duplicate IDs/texts, and invalid categories/priorities.
- **Isolation Result:** Approved records are merged into `data/raw/` for training; rejected records are saved to `data/incoming/rejected_tickets.csv` with explicit rejection reasons.

---

## 5. Reports Created in `reports/m2/`

1. `dataset_diversity.md`: Detailed audit of vocabulary diversity, sentence length metrics, and keywords.
2. `data_validation.md`: Validation rules, PII handling, and ingestion accepted/rejected counts.
3. `dataset_version.md`: Dataset version v2.0-M2 metadata and source/category/priority distributions.
4. `model_comparison.md`: Evaluation of 4 candidate classifiers for Category and Priority.
5. `realistic_holdout.md`: Performance analysis on 60 out-of-distribution realistic holdout tickets.
6. `m2_summary.md`: This executive summary document.

---

## 6. Reproduction Commands

```bash
# 1. Generate enhanced synthetic dataset and run incoming ingestion & prep pipeline
python scripts/generate_data.py

# 2. Run data validation audit
python scripts/validate_dataset.py

# 3. Retrain all candidate models
python scripts/train_models.py

# 4. Evaluate on realistic holdout set
python scripts/evaluate_holdout.py

# 5. Run test suite
python -m pytest tests/ -v

# 6. Start Flask web application
python main.py
```

---

## 7. Milestone 2 Final Status

**M2 STATUS: PASS**
