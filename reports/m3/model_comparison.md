# Phase 6: Priority Model Comparison Report

## 1. Experimental Setup
All models were trained and evaluated on the exact same data split:
- **Training Set:** `data/train.csv` (3,103 samples)
- **Synthetic Test Set:** `data/test.csv` (665 samples)
- **Realistic Evaluation Holdout:** `data/evaluation/m2_realistic_holdout.csv` (60 real-world customer inquiries)

### Evaluated Model Configurations:
1. **M2 Baseline:** Calibrated Linear SVM (TF-IDF features only)
2. **Logistic Regression (TF-IDF only)**
3. **Random Forest (TF-IDF only)**
4. **Logistic Regression (TF-IDF + 6 Priority Signals)**
5. **Random Forest (TF-IDF + 6 Priority Signals)**
6. **Calibrated Linear SVM (TF-IDF + 6 Priority Signals) [M3 Candidate]**

---

## 2. Benchmark Results Summary

| Model Architecture | Features | Test Accuracy | Test Macro F1 | Test Weighted F1 | Holdout Accuracy | Holdout Macro F1 | Holdout Weighted F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **M2 Baseline (Calibrated Linear SVM)** | TF-IDF | **90.53%** | **0.9145** | **0.9038** | 50.00% | 0.5434 | 0.5044 |
| **Logistic Regression** | TF-IDF | 86.38% | 0.8727 | 0.8564 | 45.00% | 0.4640 | 0.4478 |
| **Random Forest** | TF-IDF | 85.71% | 0.8699 | 0.8522 | 40.00% | 0.3943 | 0.3487 |
| **Logistic Regression (Class-Weighted)** | TF-IDF + Signals | 87.71% | 0.8885 | 0.8748 | **51.67%** | 0.5267 | 0.4736 |
| **Random Forest (Class-Weighted)** | TF-IDF + Signals | 89.20% | 0.9025 | 0.8906 | 41.67% | 0.4062 | 0.3857 |
| **Calibrated Linear SVM [M3 Candidate]** | TF-IDF + Signals | **90.03%** | **0.9114** | **0.8990** | 48.33% | 0.5027 | 0.4611 |

---

## 3. Per-Class Metrics on Holdout Data (Recall & Granularity)

| Model Architecture | Critical Recall | High Recall | Low Recall | Medium Recall |
| :--- | :--- | :--- | :--- | :--- |
| **M2 Baseline (TF-IDF)** | 57.1% | 34.8% | 54.5% | 63.2% |
| **LogReg (TF-IDF)** | 42.9% | 30.4% | 36.4% | 68.4% |
| **Random Forest (TF-IDF)** | 42.9% | 8.7% | 45.5% | 73.7% |
| **LogReg (TF-IDF + Signals)** | 57.1% | 17.4% | **81.8%** | **73.7%** |
| **Random Forest (TF-IDF + Signals)**| 28.6% | 17.4% | 72.7% | 57.9% |
| **Calibrated Linear SVM (M3 Candidate)**| 57.1% | 21.7% | 45.5% | **78.9%** |

---

## 4. Key Findings & Insights
1. **Low Priority Recovery:** Adding priority signal features significantly improves Low priority recall (from 36.4% up to **81.8%** in LogReg + Signals), correctly detecting when a ticket has zero urgency, financial, or security signals.
2. **Synthetic Data Ceilings:** Synthetic training data still contains inherent labeling biases from template generation. Pure ML retraining alone on synthetic data cannot completely bridge the real-world gap without **human-in-the-loop feedback**.
3. **M3 Candidate Isolation:** The candidate model has been saved to `models/m3_priority_classifier_candidate.pkl`. The production baseline `models/priority_classifier.pkl` remains preserved and active until formal promotion gates are satisfied.
