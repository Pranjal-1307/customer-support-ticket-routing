# Phase 3: Language & Multilingual Bias Analysis Report

## 1. Overview & Detection Limitations
The customer support dataset was audited to identify language diversity and analyze whether non-English / Hinglish queries suffer from systematic prediction skew.

> **Methodological Note / Limitation:** Language classification is conducted using a combined script-detection (Devanagari Unicode check) and lexical marker heuristic (top Hindi/Hinglish functional particles). On short, code-mixed customer support text (e.g., `"plz check mera order"`), token overlap creates minor boundary fuzziness. However, it provides an objective aggregate measure across the corpus.

---

## 2. Language Distribution in Training Data

| Language Group | Sample Count | Percentage | Primary Characteristics |
| :--- | :--- | :--- | :--- |
| **English / Standard** | 2,873 | 92.59% | Standard English templates and natural queries |
| **Hinglish / Romanized Hindi** | 230 | 7.41% | Romanized Hindi conversational inquiries |
| **Devanagari Script** | 0 | 0.00% | No native Devanagari script samples in base dataset |
| **Total** | 3,103 | 100.0% | |

---

## 3. Language Group Cross-Tabulations

### Language × Priority Breakdown:
| Language Group | Critical | High | Low | Medium | Total |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **English / Standard** | 383 (13.3%) | 834 (29.0%) | 596 (20.7%) | 1,060 (36.9%) | 2,873 |
| **Hinglish / Romanized Hindi** | 10 (4.3%) | 32 (13.9%) | 47 (20.4%) | **141 (61.3%)** | 230 |
| **Total** | 393 (12.7%) | 866 (27.9%) | 643 (20.7%) | 1,201 (38.7%) | 3,103 |

**Finding:** Hinglish samples in the training set are overwhelmingly labeled as **Medium (61.3%)**, with Critical priority underrepresented at only **4.3%** compared to 13.3% in English.

---

## 4. Semantic Equivalence Testing (English vs. Hinglish)

We compared the exact prediction outputs and probability distributions of semantically identical English and Hinglish support tickets on the frozen M2 production model:

### Test Pair 1: Routine Order Status
- **English:** `"Where is my order?"`
  - Category: Refund (42.37%) | Priority: **Medium** (89.76%)
  - Distribution: `{Low: 1.36%, Medium: 89.76%, High: 8.80%, Critical: 0.08%}`
- **Hinglish:** `"Mera order kahan hai?"`
  - Category: Billing (61.20%) | Priority: **Medium** (91.11%)
  - Distribution: `{Low: 1.75%, Medium: 91.11%, High: 7.01%, Critical: 0.13%}`
- **Analysis:** Both behave similarly for routine queries, defaulting to the dataset's Medium prior.

### Test Pair 2: Account Profile Update
- **English:** `"I need to update my account."`
  - Category: Account (75.48%) | Priority: **High** (67.58%)
  - Distribution: `{Low: 21.98%, Medium: 9.79%, High: 67.58%, Critical: 0.65%}`
- **Hinglish:** `"Mujhe apna account update karna hai."`
  - Category: Account (89.93%) | Priority: **High** (56.92%)
  - Distribution: `{Low: 0.69%, Medium: 42.12%, High: 56.92%, Critical: 0.28%}`
- **Analysis:** High priority is spuriously triggered on both due to the `"account"` topic token.

### Test Pair 3: Urgent Double Charge (High Urgency)
- **English:** `"My card was charged twice urgently."`
  - Category: Billing (98.36%) | Priority: **High** (65.84%)
  - Distribution: `{Low: 0.17%, Medium: 31.29%, High: 65.84%, Critical: 2.70%}`
- **Hinglish:** `"Mera card do baar charge ho gaya hai jaldi refund karo."`
  - Category: Refund (51.56%) | Priority: **Medium** (94.54%)
  - Distribution: `{Low: 0.13%, Medium: 94.54%, High: 0.50%, Critical: 4.84%}`
- **Analysis:** **Severe Bias Confirmed.** The Hinglish ticket was routed to Refund and assigned 94.54% Medium priority because the Hindi urgency markers (`"jaldi"`, `"do baar"`) were ignored by the English-only TF-IDF vocabulary.

### Test Pair 4: Hacked Account (Critical Security)
- **English:** `"My account has been hacked and credentials changed."`
  - Category: Account (92.99%) | Priority: **High** (72.51%)
  - Distribution: `{Low: 0.54%, Medium: 22.28%, High: 72.51%, Critical: 4.67%}`
- **Hinglish:** `"Mera account hack ho gaya hai."`
  - Category: Account (69.11%) | Priority: **Medium** (87.97%)
  - Distribution: `{Low: 0.36%, Medium: 87.97%, High: 6.80%, Critical: 4.87%}`
- **Analysis:** Neither model flags Critical, but the Hinglish variant degrades further to Medium (87.97%) due to missing vocabulary weights for `"hack"`.

---

## 5. Architectural Implications for M3
1. **Multilingual Signal Extraction:** Priority signal extractors must include both English and Romanized Hindi/Hinglish urgency terms (e.g. `urgent`, `asap`, `emergency`, `jaldi`, `turant`, `fraud`, `hack`, `chori`, `dhokhadhadi`).
2. **Signal Features as Normalizers:** By extracting explicit signal scores (Urgency, Security Risk, Financial Impact), both English and Hinglish expressions map to the same continuous feature dimensions, mitigating TF-IDF vocabulary sparseness.
