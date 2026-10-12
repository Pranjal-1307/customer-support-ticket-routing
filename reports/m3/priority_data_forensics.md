# Phase 2: Priority Data Forensics Report

## 1. Executive Summary
An exhaustive data forensic audit was conducted on `data/train.csv` (3,103 samples), `data/validation.csv` (665 samples), and `data/test.csv` (665 samples). 

The audit reveals why the M2 priority model performs well on synthetic in-distribution test sets (89.5% accuracy) but drops to **51.67%** on realistic holdout queries. The root cause is severe **lexical-topic confounding** and **vocabulary gaps** for realistic priority expressions.

---

## 2. Priority Class Distribution (Training Set)

| Priority Class | Sample Count | Percentage | Class Balance Assessment |
| :--- | :--- | :--- | :--- |
| **Low** | 643 | 20.72% | Underrepresented |
| **Medium** | 1,201 | 38.70% | Dominant class |
| **High** | 866 | 27.91% | Well represented |
| **Critical** | 393 | 12.67% | Severely underrepresented (1:3 ratio with Medium) |
| **Total** | 3,103 | 100.0% | |

---

## 3. Category × Priority Cross-Tabulation Matrix (Train)

| Category | Critical | High | Low | Medium | Total | Primary Priority Distribution |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Account** | 78 (20.6%) | 129 (34.0%) | 71 (18.7%) | 101 (26.6%) | 379 | Balanced across 4 classes |
| **Billing** | 69 (16.9%) | 86 (21.1%) | 60 (14.7%) | 193 (47.3%) | 408 | Skewed towards Medium |
| **Cancellation** | 0 (0.0%) | 102 (27.5%) | 114 (30.7%) | 155 (41.8%) | 371 | **0 Critical samples** |
| **Complaint** | 73 (21.3%) | 99 (28.9%) | 63 (18.4%) | 107 (31.3%) | 342 | Distributed |
| **Product Inquiry** | 0 (0.0%) | 53 (14.0%) | 235 (62.2%) | 90 (23.8%) | 378 | **0 Critical samples**, 62.2% Low |
| **Refund** | 0 (0.0%) | 132 (30.3%) | 26 (6.0%) | 277 (63.7%) | 435 | **0 Critical samples**, 63.7% Medium |
| **Shipping** | 31 (8.0%) | 176 (45.5%) | 74 (19.1%) | 106 (27.4%) | 387 | 45.5% High |
| **Technical Support**| 142 (35.2%) | 89 (22.1%) | 0 (0.0%) | 172 (42.7%) | 403 | **0 Low samples**, 35.2% Critical |
| **Total** | **393** | **866** | **643** | **1,201** | **3,103** | |

### Structural Imbalances Identified:
1. **Zero-Probability Slices:** In synthetic data, `Cancellation`, `Product Inquiry`, and `Refund` have **0% Critical tickets**. `Technical Support` has **0% Low tickets**.
2. **Artificial Priors:** A ticket mentioning "refund" is biased towards Medium (63.7%), while a ticket mentioning "shipping" or "parcel" is biased towards High (45.5%).

---

## 4. Word and Bigram Analysis per Priority

### Priority Vocabulary Signatures:
- **Low Priority:**
  - Top Words: `help` (165), `plz` (157), `quick` (155), `email` (148), `support` (131), `contact` (107).
  - Top Bigrams: `contact via` (106), `via email` (106), `kindly look` (103), `into matter` (103), `looking forward` (91).
- **Medium Priority:**
  - Top Words: `plz` (264), `order` (263), `help` (254), `quick` (241), `possible` (234), `account` (229), `soon` (225), `refund` (215).
  - Top Bigrams: `advise soon` (225), `soon possible` (225), `urgent attention` (158), `hi support` (153), `quick question` (132).
- **High Priority:**
  - Top Words: `urgent` (462), `plz` (189), `order` (182), `help` (164), `account` (160), `refund` (147), `package` (144).
  - Top Bigrams: `urgent attention` (135), `attention required` (135), `plz respond` (112), `respond asap` (112), `appreciate quick` (96).
- **Critical Priority:**
  - Top Words: `team` (102), `error` (92), `service` (92), `server` (91), `urgent` (90), `customer` (76).
  - Top Bigrams: `credit card` (68), `server error` (62), `plz respond` (45), `respond asap` (45).

---

## 5. Target Keyword Diagnostic: Topic (Category) vs. Urgency (Priority Signal)

| Keyword | Count (Train) | Dominant Priority | Priority Distribution | Dominant Category | Signal vs. Topic Classification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **parcel** | 120 | **High** (84.2%) | High: 101, Low: 16, Med: 3 | Shipping (100%) | **B. Topic Token** (Spuriously correlated with High) |
| **delivery** | 121 | **Medium** (47.1%)| Med: 57, High: 54, Low: 10 | Shipping (100%) | **B. Topic Token** |
| **shipping** | 96 | **High** (80.2%) | High: 77, Med: 18, Low: 1 | Shipping (54%), Refund (46%) | **B. Topic Token** |
| **tracking** | 133 | **High** (34.6%) | High: 46, Low: 45, Med: 42 | Shipping (99%) | **B. Topic Token** |
| **refund** | 388 | **Medium** (55.4%)| Med: 215, High: 147, Low: 26 | Refund (93%) | **B. Topic Token** (Spuriously drags tickets to Med) |
| **charged** | 94 | **Medium** (54.3%)| Med: 51, Low: 29, High: 13, Crit: 1 | Billing (54%), Cancel (46%) | **B. Topic Token** |
| **urgent** | 656 | **High** (63.7%) | High: 418, Med: 158, Crit: 80 | Uniform across all categories | **A. True Urgency Signal** |
| **asap** | 336 | **High** (33.3%) | High: 112, Med: 104, Low: 75, Crit: 45 | Uniform across all categories | **A. True Urgency Signal** |
| **today** | 31 | **Medium** (87.1%)| Med: 27, High: 4 | Cancellation (100%) | **A. Time Sensitivity Signal** |
| **hacked** | **0** | **None** | **0 in Train** | None | **A. Security Risk Signal** (Out-of-vocabulary gap) |
| **password** | 77 | **Low** (50.6%) | Low: 39, Crit: 18, High: 13, Med: 7 | Account (100%) | **B. Topic Token** |
| **account** | 465 | **Medium** (49.2%)| Med: 229, High: 160, Low: 44, Crit: 32 | Account (36%), Cancel (32%) | **B. Topic Token** |
| **delayed** | **0** | **None** | **0 in Train** | None | **A. Time/Service Impact Signal** (Out-of-vocabulary gap) |
| **missing** | 68 | **Critical** (45.6%)| Crit: 31, Med: 29, High: 8 | Cancel (53%), Shipping (47%) | **A/B. Mixed Signal** |
| **broken** | 34 | **High** (100.0%)| High: 34 | Refund (100%) | **A. Severity Signal** |

---

## 6. Root Cause Conclusion & M3 Solution
- **The Failure Mode:** The TF-IDF priority classifier is learning *category vocabulary* rather than *urgency semantics*. A customer asking `"Where is my parcel?"` receives High priority because `"parcel"` only co-occurred with High priority templates in the synthetic dataset. Conversely, `"My account was hacked"` gets only 0.06% Critical probability because the word `"hacked"` never appeared in training.
- **The M3 Solution:** Decouple domain topics from urgency dynamics by introducing **Interpretable Multi-Dimensional Priority Signals** (Urgency, Severity, Financial Impact, Security Risk, Service Impact, Time Sensitivity) as dedicated engineered features alongside TF-IDF.
