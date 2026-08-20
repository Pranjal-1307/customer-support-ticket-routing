# Data Diversity Audit Report — Milestone 2

**Dataset Analyzed:** `data/processed/cleaned_tickets.csv`  
**Total Unique Records:** 4341  

---

## 1. Ticket Text Length & Structural Distribution

| Metric | Word Count | Character Count |
|---|---|---|
| **Mean** | 19.5 words | 122.2 chars |
| **Median** | 18 words | 108 chars |
| **Min** | 2 words | 14 chars |
| **Max** | 53 words | 336 chars |

### Length Categorization
- **Short Tickets (≤6 words):** 123 (2.8%) — *e.g., "Charged twice for order #99201."*
- **Medium Tickets (7–20 words):** 2538 (58.5%) — *e.g., "Why am I seeing two charges for the same order on my bank statement?"*
- **Long Tickets (>20 words):** 1680 (38.7%) — *e.g., Detailed multi-sentence incident descriptions with background context.*

---

## 2. Vocabulary & Token Diversity

- **Total Processed Word Tokens:** 82,806
- **Unique Vocabulary Size:** 1,051 distinct terms
- **Type-Token Ratio (TTR):** 0.0127

### Comparison: M1 Baseline vs M2 Realistic Dataset
- **M1 Baseline:** ~238 template pattern variations, low vocabulary diversity, fixed phrase structures.
- **M2 Realistic Dataset:** 4341 distinct tickets across short/medium/long formats, conversational phrasing, questions, complaints, subtle spelling variations, and multilingual terms.

---

## 3. Multilingual & Code-Mixed Language Representation

| Language / Dialect | Count | Percentage | Pipeline Compatibility |
|---|---|---|---|
| **English (Standard/Informal)** | 2644 | 60.9% | Full TF-IDF compatibility |
| **Hinglish / Hindi Code-Mixed** | 1487 | 34.3% | ASCII subword/character tokenization |
| **Gujarati Code-Mixed** | 210 | 4.8% | ASCII transliterated tokenization |

*Note: Multilingual support is provided via transliterated code-mixed text compatible with standard TF-IDF unigram/bigram tokenization. Full native script NLU (e.g. Devanagari script) is out of scope for M2.*

---

## 4. Category-Specific & Priority-Specific Keywords

### Top Discriminative Terms per Category:
- **Technical Support:** `to, the, on, error, this, help, a, database` 
- **Cancellation:** `please, to, account, cancellation, cancel, i, my, for` 
- **Complaint:** `for, on, service, support, the, ticket, this, urgent` 
- **Billing:** `for, on, the, this, of, to, my, a` 
- **Product Inquiry:** `for, a, the, and, on, support, team, does` 
- **Shipping:** `the, on, my, package, order, delivery, tracking, for` 
- **Refund:** `refund, order, the, to, for, i, a, this` 
- **Account:** `account, i, the, this, help, email, password, on` 

---

## 5. Main Weaknesses Identified & M2 Mitigations

1. **Category Over-Separability in M1:** M1 relied on category-exclusive keywords (e.g. "refund", "shipping") causing artificial 97%+ accuracy. **M2 Mitigation:** Introduced indirect wording (e.g., "Why is there a duplicate line item on my statement?" instead of "I want a refund") and cross-category vocabulary overlap.
2. **Template Dependence:** M1 dataset was heavily grouped around 80 template patterns. **M2 Mitigation:** Expanded pattern pool, added short/medium/long variations, conversational prefixes, and informal noise.
3. **Priority Keyword Rigidity:** M1 priority was coupled directly to synthetic urgency suffixes. **M2 Mitigation:** Implemented rule-driven deterministic priority assignment based on domain impact (financial security, server downtime, low-impact inquiries) independent of single fixed keywords.
