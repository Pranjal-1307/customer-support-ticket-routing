"""
Phase 2: Data Forensics Analysis Script for M3
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import Counter
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.preprocess import preprocess_text

def run_data_forensics():
    train_path = os.path.join(BASE_DIR, "data", "train.csv")
    val_path = os.path.join(BASE_DIR, "data", "validation.csv")
    test_path = os.path.join(BASE_DIR, "data", "test.csv")
    holdout_path = os.path.join(BASE_DIR, "data", "evaluation", "m2_realistic_holdout.csv")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    df_holdout = pd.read_csv(holdout_path) if os.path.exists(holdout_path) else None

    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

    print("=== TRAIN DATASET PRIORITY COUNTS ===")
    pri_counts = df_train['priority'].value_counts()
    pri_pcts = df_train['priority'].value_counts(normalize=True) * 100
    for p in ['Low', 'Medium', 'High', 'Critical']:
        cnt = pri_counts.get(p, 0)
        pct = pri_pcts.get(p, 0.0)
        print(f"  {p}: {cnt} ({pct:.2f}%)")

    print("\n=== CATEGORY x PRIORITY CROSS-TABULATION (Train) ===")
    ct = pd.crosstab(df_train['category'], df_train['priority'], margins=True)
    print(ct)

    # Word & Bigram analysis per priority
    stopwords = {'i', 'my', 'the', 'a', 'an', 'is', 'to', 'for', 'and', 'in', 'of', 'it', 'me', 'please', 'this', 'was', 'with', 'on', 'have', 'be', 'you', 'can', 'not', 'do', 'at', 'but', 'that', 'as', 'are', 'we', 'they', 'our', 'from'}

    def get_ngrams(texts, n=1):
        ngrams_list = []
        for t in texts:
            words = [w for w in re.findall(r'\b[a-z]{2,}\b', str(t).lower()) if w not in stopwords]
            if n == 1:
                ngrams_list.extend(words)
            elif n == 2:
                for i in range(len(words)-1):
                    ngrams_list.append(f"{words[i]} {words[i+1]}")
        return Counter(ngrams_list)

    print("\n=== TOP WORDS PER PRIORITY ===")
    top_words_by_pri = {}
    top_bigrams_by_pri = {}
    for p in ['Low', 'Medium', 'High', 'Critical']:
        sub = df_train[df_train['priority'] == p]['ticket_text']
        w_cnt = get_ngrams(sub, n=1)
        bg_cnt = get_ngrams(sub, n=2)
        top_words_by_pri[p] = w_cnt.most_common(10)
        top_bigrams_by_pri[p] = bg_cnt.most_common(10)
        print(f"\nPriority: {p}")
        print(f"  Top Words: {top_words_by_pri[p]}")
        print(f"  Top Bigrams: {top_bigrams_by_pri[p]}")

    # Specific Target Words Investigation
    target_words = [
        "parcel", "delivery", "shipping", "tracking", "refund", "charged", 
        "urgent", "asap", "today", "hacked", "password", "account", 
        "delayed", "missing", "broken"
    ]

    word_analysis = []
    print("\n=== TARGET WORD INVESTIGATION ===")
    for tw in target_words:
        matched = df_train[df_train['ticket_text'].str.contains(r'\b' + tw + r'\b', case=False, na=False)]
        total_occurrences = len(matched)
        if total_occurrences > 0:
            pri_dist = matched['priority'].value_counts().to_dict()
            cat_dist = matched['category'].value_counts().to_dict()
            dominant_pri = matched['priority'].mode()[0] if not matched.empty else "N/A"
            dominant_cat = matched['category'].mode()[0] if not matched.empty else "N/A"
        else:
            pri_dist, cat_dist, dominant_pri, dominant_cat = {}, {}, "None", "None"
        
        word_analysis.append({
            "word": tw,
            "count": total_occurrences,
            "pri_dist": pri_dist,
            "cat_dist": cat_dist,
            "dominant_pri": dominant_pri,
            "dominant_cat": dominant_cat
        })
        print(f"Word: '{tw}' (Count: {total_occurrences}) -> Dominant Priority: {dominant_pri} | Priority Dist: {pri_dist} | Category Dist: {cat_dist}")

    # Sample tickets per priority
    samples_by_pri = {}
    for p in ['Low', 'Medium', 'High', 'Critical']:
        samples_by_pri[p] = df_train[df_train['priority'] == p]['ticket_text'].head(4).tolist()

    return {
        "pri_counts": pri_counts.to_dict(),
        "pri_pcts": pri_pcts.to_dict(),
        "cross_tab": ct,
        "top_words": top_words_by_pri,
        "top_bigrams": top_bigrams_by_pri,
        "word_analysis": word_analysis,
        "samples": samples_by_pri
    }

if __name__ == "__main__":
    run_data_forensics()
