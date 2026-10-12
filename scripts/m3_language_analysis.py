"""
Phase 3: Language Bias Investigation for M3
"""

import os
import sys
import pandas as pd
import numpy as np
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.predict import get_predictor
from utils.preprocess import preprocess_text

def detect_language(text):
    """
    Lightweight rule-based language heuristic for English vs Hindi (Devanagari) vs Hinglish.
    Note: Heuristic classification has limitations on short, mixed-code texts.
    """
    text_str = str(text).strip()
    # Check for Devanagari script
    if re.search(r'[\u0900-\u097F]', text_str):
        return "Hindi (Devanagari)"
    
    # Common Hinglish/Romanized Hindi stopwords & markers
    hinglish_markers = {
        'mera', 'meri', 'mere', 'kahan', 'hai', 'mujhe', 'karna', 'karo', 'apna', 'apni',
        'kya', 'nahi', 'raha', 'rahi', 'kaise', 'kab', 'aur', 'yeh', 'woh', 'bhai', 'ji',
        'chahiye', 'hoga', 'gaya', 'paise', 'kat', 'gaye', 'baat', 'samay', 'jaldi', 'madad'
    }
    tokens = set(re.findall(r'\b[a-zA-Z]{2,}\b', text_str.lower()))
    hinglish_match_count = len(tokens.intersection(hinglish_markers))
    
    if hinglish_match_count >= 1:
        return "Hinglish / Romanized Hindi"
    return "English / Other"

def run_language_analysis():
    train_path = os.path.join(BASE_DIR, "data", "train.csv")
    df_train = pd.read_csv(train_path)

    df_train['detected_lang'] = df_train['ticket_text'].apply(detect_language)

    print("=== LANGUAGE DISTRIBUTION IN TRAIN DATASET ===")
    print(df_train['detected_lang'].value_counts())
    print("\n=== LANGUAGE x PRIORITY CROSS-TABULATION ===")
    print(pd.crosstab(df_train['detected_lang'], df_train['priority'], margins=True))

    print("\n=== LANGUAGE x CATEGORY CROSS-TABULATION ===")
    print(pd.crosstab(df_train['detected_lang'], df_train['category'], margins=True))

    # Test Semantic Pairs on Current Production Model
    predictor = get_predictor()
    classes = list(predictor.priority_classifier.classes_)
    
    pairs = [
        ("English: Order Status", "Where is my order?", "Hinglish: Order Status", "Mera order kahan hai?"),
        ("English: Account Update", "I need to update my account.", "Hinglish: Account Update", "Mujhe apna account update karna hai."),
        ("English: Urgent Charge", "My card was charged twice urgently.", "Hinglish: Urgent Charge", "Mera card do baar charge ho gaya hai jaldi refund karo."),
        ("English: Hacked Account", "My account has been hacked and credentials changed.", "Hinglish: Hacked Account", "Mera account hack ho gaya hai.")
    ]

    print("\n=== SEMANTIC PAIR EVALUATION ON M2 PRODUCTION MODEL ===")
    pair_results = []
    for en_label, en_text, hi_label, hi_text in pairs:
        # EN prediction
        res_en = predictor.predict(en_text)
        vec_en = predictor.vectorizer.transform([preprocess_text(en_text)])
        pri_probs_en = {cls: round(float(p)*100, 2) for cls, p in zip(classes, predictor.priority_classifier.predict_proba(vec_en)[0])}
        
        # HI prediction
        res_hi = predictor.predict(hi_text)
        vec_hi = predictor.vectorizer.transform([preprocess_text(hi_text)])
        pri_probs_hi = {cls: round(float(p)*100, 2) for cls, p in zip(classes, predictor.priority_classifier.predict_proba(vec_hi)[0])}

        print(f"\n--- Pair: {en_label} vs {hi_label} ---")
        print(f"EN: '{en_text}'")
        print(f"  Category: {res_en['category']} ({res_en['category_confidence']}%) | Priority: {res_en['priority']} ({res_en['priority_confidence']}%)")
        print(f"  Priority Distribution: {pri_probs_en}")
        print(f"HI: '{hi_text}'")
        print(f"  Category: {res_hi['category']} ({res_hi['category_confidence']}%) | Priority: {res_hi['priority']} ({res_hi['priority_confidence']}%)")
        print(f"  Priority Distribution: {pri_probs_hi}")

        pair_results.append({
            "pair": f"{en_label} / {hi_label}",
            "en": {"text": en_text, "cat": res_en['category'], "pri": res_en['priority'], "dist": pri_probs_en},
            "hi": {"text": hi_text, "cat": res_hi['category'], "pri": res_hi['priority'], "dist": pri_probs_hi}
        })

    return pair_results

if __name__ == "__main__":
    run_language_analysis()
