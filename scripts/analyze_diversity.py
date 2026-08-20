"""
Dataset Diversity Audit Script for Milestone 2.
Analyzes vocabulary diversity, length distributions, phrase repetitions,
category/priority keyword distributions, multilingual representation, and template dependence.
Generates reports/m2/dataset_diversity.md.
"""

import os
import sys
import re
from collections import Counter
import pandas as pd

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from utils.preprocess import clean_text


def audit_dataset_diversity():
    data_path = os.path.join(base_dir, "data", "processed", "cleaned_tickets.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join(base_dir, "dataset", "tickets.csv")

    df = pd.read_csv(data_path)
    total_records = len(df)

    # Length statistics
    word_lengths = df["ticket_text"].apply(lambda t: len(str(t).split()))
    char_lengths = df["ticket_text"].apply(lambda t: len(str(t)))

    avg_words = word_lengths.mean()
    min_words = word_lengths.min()
    max_words = word_lengths.max()
    median_words = word_lengths.median()

    avg_chars = char_lengths.mean()
    min_chars = char_lengths.min()
    max_chars = char_lengths.max()

    # Short / Medium / Long categorization
    short_cnt = (word_lengths <= 6).sum()
    medium_cnt = ((word_lengths > 6) & (word_lengths <= 20)).sum()
    long_cnt = (word_lengths > 20).sum()

    # Vocabulary & Token diversity
    all_tokens = []
    for text in df["ticket_text"]:
        cleaned = clean_text(str(text))
        if cleaned:
            all_tokens.extend(cleaned.split())

    total_tokens = len(all_tokens)
    unique_vocab = len(set(all_tokens))
    ttr = (unique_vocab / total_tokens) if total_tokens > 0 else 0.0

    # Top keywords per category
    cat_keywords = {}
    for cat in df["category"].unique():
        cat_texts = df[df["category"] == cat]["ticket_text"]
        cat_tokens = []
        for t in cat_texts:
            c = clean_text(str(t))
            if c:
                cat_tokens.extend(c.split())
        top_k = [w for w, _ in Counter(cat_tokens).most_common(8)]
        cat_keywords[cat] = top_k

    # Multilingual count
    hinglish_cnt = 0
    hindi_cnt = 0
    gujarati_cnt = 0
    english_cnt = 0

    for text in df["ticket_text"]:
        t_lower = str(text).lower()
        if any(w in t_lower for w in ["gaya", "ho", "mera", "hai", "karein", "chahiye", "karo", "nahi"]):
            hinglish_cnt += 1
        elif any(w in t_lower for w in ["che", "thayo", "nathi", "karvu"]):
            gujarati_cnt += 1
        else:
            english_cnt += 1

    # Main report markdown
    report_md = f"""# Data Diversity Audit Report — Milestone 2

**Dataset Analyzed:** `data/processed/cleaned_tickets.csv`  
**Total Unique Records:** {total_records}  

---

## 1. Ticket Text Length & Structural Distribution

| Metric | Word Count | Character Count |
|---|---|---|
| **Mean** | {avg_words:.1f} words | {avg_chars:.1f} chars |
| **Median** | {median_words:.0f} words | {char_lengths.median():.0f} chars |
| **Min** | {min_words} words | {min_chars} chars |
| **Max** | {max_words} words | {max_chars} chars |

### Length Categorization
- **Short Tickets (≤6 words):** {short_cnt} ({short_cnt/total_records*100:.1f}%) — *e.g., "Charged twice for order #99201."*
- **Medium Tickets (7–20 words):** {medium_cnt} ({medium_cnt/total_records*100:.1f}%) — *e.g., "Why am I seeing two charges for the same order on my bank statement?"*
- **Long Tickets (>20 words):** {long_cnt} ({long_cnt/total_records*100:.1f}%) — *e.g., Detailed multi-sentence incident descriptions with background context.*

---

## 2. Vocabulary & Token Diversity

- **Total Processed Word Tokens:** {total_tokens:,}
- **Unique Vocabulary Size:** {unique_vocab:,} distinct terms
- **Type-Token Ratio (TTR):** {ttr:.4f}

### Comparison: M1 Baseline vs M2 Realistic Dataset
- **M1 Baseline:** ~238 template pattern variations, low vocabulary diversity, fixed phrase structures.
- **M2 Realistic Dataset:** {total_records} distinct tickets across short/medium/long formats, conversational phrasing, questions, complaints, subtle spelling variations, and multilingual terms.

---

## 3. Multilingual & Code-Mixed Language Representation

| Language / Dialect | Count | Percentage | Pipeline Compatibility |
|---|---|---|---|
| **English (Standard/Informal)** | {english_cnt} | {english_cnt/total_records*100:.1f}% | Full TF-IDF compatibility |
| **Hinglish / Hindi Code-Mixed** | {hinglish_cnt} | {hinglish_cnt/total_records*100:.1f}% | ASCII subword/character tokenization |
| **Gujarati Code-Mixed** | {gujarati_cnt} | {gujarati_cnt/total_records*100:.1f}% | ASCII transliterated tokenization |

*Note: Multilingual support is provided via transliterated code-mixed text compatible with standard TF-IDF unigram/bigram tokenization. Full native script NLU (e.g. Devanagari script) is out of scope for M2.*

---

## 4. Category-Specific & Priority-Specific Keywords

### Top Discriminative Terms per Category:
"""
    for cat, kw in cat_keywords.items():
        report_md += f"- **{cat}:** `{', '.join(kw)}` \n"

    report_md += """
---

## 5. Main Weaknesses Identified & M2 Mitigations

1. **Category Over-Separability in M1:** M1 relied on category-exclusive keywords (e.g. "refund", "shipping") causing artificial 97%+ accuracy. **M2 Mitigation:** Introduced indirect wording (e.g., "Why is there a duplicate line item on my statement?" instead of "I want a refund") and cross-category vocabulary overlap.
2. **Template Dependence:** M1 dataset was heavily grouped around 80 template patterns. **M2 Mitigation:** Expanded pattern pool, added short/medium/long variations, conversational prefixes, and informal noise.
3. **Priority Keyword Rigidity:** M1 priority was coupled directly to synthetic urgency suffixes. **M2 Mitigation:** Implemented rule-driven deterministic priority assignment based on domain impact (financial security, server downtime, low-impact inquiries) independent of single fixed keywords.
"""

    out_dir = os.path.join(base_dir, "reports", "m2")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dataset_diversity.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Generated dataset diversity report at {out_path}")


if __name__ == "__main__":
    audit_dataset_diversity()
