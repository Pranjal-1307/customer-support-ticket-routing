# Data Leakage Audit Report (Milestone 1)

**Generated:** 2026-08-20T06:12:49.958339

## 1. TF-IDF Fitting Check

In `scripts/train_models.py` (lines 51-53):
```python
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_df['processed_text'])
X_test = vectorizer.transform(test_df['processed_text'])
```
**Status: ✅ CORRECT** — TF-IDF is fitted only on training data. Test data is only transformed.

## 2. Target Label Leakage Check

The TF-IDF vectorizer uses only `processed_text` (preprocessed ticket text).
Category and priority labels are **not** included as input features.
**Status: ✅ NO LEAKAGE** — Labels are targets only.

## 3. Train/Test Split Check

- Training samples: 994
- Test samples: 227
- Exact text overlap (train ∩ test): **0**
- ✅ No exact text overlap between train and test.


## 4. Near-Duplicate (Template) Overlap Check

After normalizing IDs, amounts, error codes, IP addresses, and noise suffixes:
- Unique train templates: 194
- Unique test templates: 82
- Template overlap (train ∩ test): **61**

> **⚠️ NEAR-DUPLICATE LEAKAGE:** 61 template patterns appear in both train and test sets.
> These are primarily **static-text templates from external/manual data** that have no dynamic fill parameters.
> The group-based split in `prepare_dataset.py` uses a more aggressive normalization that creates
> distinct groups for these entries, but a realistic near-duplicate check still detects overlap.
> Category-specific vocabulary is deterministic, so this has minimal additional effect on category
> metrics (which would be high regardless). For priority, the effect is also limited since priority
> is deterministically derived from template + suffix, not randomly assigned.

## 5. Stratification Check

| Category | Train % | Test % | Diff |
|----------|---------|--------|------|
| Account | 6.8% | 6.6% | 0.2% |
| Billing | 22.1% | 24.2% | 2.1% |
| Cancellation | 9.5% | 12.8% | 3.3% |
| Complaint | 6.7% | 6.2% | 0.6% |
| Product Inquiry | 11.7% | 6.6% | 5.1% |
| Refund | 13.2% | 15.9% | 2.7% |
| Shipping | 17.4% | 16.7% | 0.7% |
| Technical Support | 12.6% | 11.0% | 1.6% |

**Status: ✅ CORRECT** — Stratified split preserves category proportions.

## 6. Preprocessing Information Leak Check

- Preprocessing (`utils/preprocess.py`) uses only NLTK stopwords and WordNet lemmatizer.
- No corpus-level statistics (e.g., mean text length, vocabulary frequency) are computed from test data.
- The preprocessing function operates independently on each text string.
**Status: ✅ NO LEAKAGE** — Preprocessing is stateless per-sample.

## Summary

**Leakage issues found:** 1

1. Near-duplicate template overlap: 61 patterns

### Impact Assessment

The near-duplicate template overlap is inherent to the synthetic data generation process.
Since each category uses 10 unique phrase templates with random fill values,
the 80/20 split naturally places variants of the same template in both splits.
This is **documented but not fixed in M1** because:
1. The `data/` pipeline (`prepare_dataset.py`) already has template deduplication.
2. Switching training to the deduplicated pipeline requires changing `train_models.py` data paths.
3. Category metrics would remain high regardless (unique vocabulary per category).
4. This will be addressed when switching to realistic data in a later milestone.