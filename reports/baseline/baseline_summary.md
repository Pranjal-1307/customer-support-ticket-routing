# Milestone 1 — Baseline Summary Report

**Generated:** 2026-08-20T06:13:18.586144

---

## Category Model Results

| Model | Accuracy | Precision | Recall | F1 | Macro F1 | Weighted F1 |
|-------|----------|-----------|--------|------|----------|-------------|
| Logistic Regression | 0.9692 | 0.9697 | 0.9692 | 0.9691 | 0.9676 | 0.9691 |
| Naive Bayes | 0.9604 | 0.9626 | 0.9604 | 0.96 | 0.9555 | 0.96 |
| Random Forest | 0.9736 | 0.9749 | 0.9736 | 0.9738 | 0.9701 | 0.9738 |
| Linear SVM | 0.9692 | 0.9697 | 0.9692 | 0.9691 | 0.9644 | 0.9691 |

**Best Model:** Random Forest (Weighted F1: 0.9738)

## Priority Model Results

| Model | Accuracy | Precision | Recall | F1 | Macro F1 | Weighted F1 |
|-------|----------|-----------|--------|------|----------|-------------|
| Logistic Regression | 0.7137 | 0.7538 | 0.7137 | 0.7087 | 0.6913 | 0.7087 |

## Category Performance Analysis

Category models achieve near-perfect scores (>95%) because:
1. **Deterministic vocabulary:** Each category uses 10 unique phrase templates with category-exclusive keywords.
2. **No vocabulary overlap:** Words like 'refund', 'billing', 'cancel', 'shipping', 'tracking' appear exclusively in their category.
3. **Template repetition:** 5200 samples from 80 templates = ~65 variants per template. TF-IDF easily learns the discriminative keywords.
4. **Near-duplicate train/test contamination:** Template variants appear in both splits, further inflating scores.
5. **This does NOT mean the model will perform at 99%+ on real customer tickets** — real tickets have diverse vocabulary, ambiguous language, and cross-category concepts.

## Priority Performance Analysis

Priority accuracy is ~71%, above the 25% random baseline for 4 classes, because:
1. **Deterministic priority labeling:** Priority is derived from template base-priority + urgency suffix upgrade, providing partial textual signal.
2. **Class imbalance:** Medium/High dominate, with Critical and Low underrepresented.
3. **Coarse features:** TF-IDF captures category/template vocabulary but not fine-grained urgency cues.
4. **This is expected M1 baseline behavior.** Further improvement requires richer text features or priority-specific engineering.

## Data Leakage Summary

**Findings:** 1 issues documented:
- Near-duplicate template overlap: 61 patterns

See `leakage_audit.md` for full details. No code fix applied in M1 (documented only).

## Reports Generated

- `reports/baseline/baseline_summary.md` (3,745 bytes)
- `reports/baseline/category_classification_report.txt` (6,729 bytes)
- `reports/baseline/category_confusion_matrix.png` (72,608 bytes)
- `reports/baseline/confidence_audit.md` (3,331 bytes)
- `reports/baseline/dataset_distribution.csv` (709 bytes)
- `reports/baseline/dataset_summary.md` (3,235 bytes)
- `reports/baseline/future_data_architecture.md` (5,142 bytes)
- `reports/baseline/leakage_audit.md` (3,480 bytes)
- `reports/baseline/model_comparison.csv` (286 bytes)
- `reports/baseline/preprocessing_audit.md` (2,429 bytes)
- `reports/baseline/priority_classification_report.txt` (3,733 bytes)
- `reports/baseline/priority_confusion_matrix.png` (42,233 bytes)

## Commands to Reproduce

```bash
# Generate dataset (if not present)
python scripts/generate_data.py

# Run complete baseline audit
python scripts/run_baseline_audit.py

# Run tests
python -m pytest tests/test_baseline.py -v

# Start application
python main.py
```

## Milestone 1 Status

**PASS WITH NOTES**

Notes:
- Near-duplicate template overlap exists in train/test split (documented, not fixed in M1).
- Priority model performs at near-random levels (expected with synthetic data).
- LinearSVC softmax confidence is not a calibrated probability (documented).
- No model versioning exists yet (documented for future milestones).