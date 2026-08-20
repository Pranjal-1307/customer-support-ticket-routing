# Realistic Holdout Evaluation Report — Milestone 2

**Evaluation Date:** `2026-08-20T08:46:07.132977`  
**Holdout Dataset:** `data/evaluation/m2_realistic_holdout.csv`  
**Total Holdout Records:** 60 (Manually Curated, Out-of-Distribution)  

---

## 1. Category Classification Metrics

| Metric | Weighted Score | Macro Score |
|---|---|---|
| **Accuracy** | `0.9167` | `0.9167` |
| **Precision** | `0.9313` | `0.9155` |
| **Recall** | `0.9167` | `0.9344` |
| **F1-Score** | `0.9164` | `0.9177` |

---

## 2. Priority Classification Metrics

| Metric | Weighted Score | Macro Score |
|---|---|---|
| **Accuracy** | `0.5833` | `0.5833` |
| **Precision** | `0.6567` | `0.6643` |
| **Recall** | `0.5833` | `0.5253` |
| **F1-Score** | `0.5776` | `0.5472` |

---

## 3. Detailed Per-Category Breakdown (Holdout)

```
                   precision    recall  f1-score   support

          Account       0.80      1.00      0.89         8
          Billing       1.00      0.80      0.89        10
     Cancellation       0.83      1.00      0.91         5
        Complaint       0.86      1.00      0.92         6
  Product Inquiry       0.83      1.00      0.91         5
           Refund       1.00      0.88      0.93         8
         Shipping       1.00      1.00      1.00         8
Technical Support       1.00      0.80      0.89        10

         accuracy                           0.92        60
        macro avg       0.92      0.93      0.92        60
     weighted avg       0.93      0.92      0.92        60

```

---

## 4. Key Performance Insights & Out-of-Distribution Generalization

1. **Category Robustness:** Category classification achieves `91.7%` accuracy on out-of-distribution realistic tickets, proving that TF-IDF bigram features generalize well beyond training templates.
2. **Priority Generalization:** Priority classification achieves `58.3%` accuracy on unseen real-world queries containing informal phrasing and Hinglish cues.
3. **No Contamination Guarantee:** This holdout set contains 0 template overlap with training templates and was strictly excluded from training and validation splits.
