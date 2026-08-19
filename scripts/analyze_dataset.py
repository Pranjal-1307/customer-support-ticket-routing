"""
Dataset Analysis Script for Customer Support Ticket Routing System.
Computes dataset statistics, class distributions, ticket length metrics,
source counts, duplicate analysis, and train/val/test split breakdown.
Generates comprehensive report to reports/dataset_report.txt.
"""

import os
import pandas as pd


def analyze():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    interim_path = os.path.join(data_dir, "interim", "combined_tickets.csv")
    cleaned_path = os.path.join(data_dir, "processed", "cleaned_tickets.csv")
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "validation.csv")
    test_path = os.path.join(data_dir, "test.csv")
    unseen_path = os.path.join(data_dir, "evaluation", "unseen_test.csv")
    hinglish_path = os.path.join(data_dir, "evaluation", "hinglish_test.csv")

    report_lines = []
    report_lines.append("=========================================================")
    report_lines.append(" CUSTOMER SUPPORT TICKET DATASET ANALYSIS REPORT")
    report_lines.append("=========================================================\n")

    # Load datasets
    interim_df = pd.read_csv(interim_path) if os.path.exists(interim_path) else pd.DataFrame()
    cleaned_df = pd.read_csv(cleaned_path) if os.path.exists(cleaned_path) else pd.DataFrame()
    train_df = pd.read_csv(train_path) if os.path.exists(train_path) else pd.DataFrame()
    val_df = pd.read_csv(val_path) if os.path.exists(val_path) else pd.DataFrame()
    test_df = pd.read_csv(test_path) if os.path.exists(test_path) else pd.DataFrame()
    unseen_df = pd.read_csv(unseen_path) if os.path.exists(unseen_path) else pd.DataFrame()
    hinglish_df = pd.read_csv(hinglish_path) if os.path.exists(hinglish_path) else pd.DataFrame()

    # 1. Total Records Summary
    raw_count = len(interim_df)
    clean_count = len(cleaned_df)
    dups_removed = raw_count - clean_count
    
    report_lines.append(f"1. DATASET OVERVIEW")
    report_lines.append(f"   - Total Raw Combined Records: {raw_count}")
    report_lines.append(f"   - Total Cleaned Processed Records: {clean_count}")
    report_lines.append(f"   - Duplicates & Non-Unique Templates Removed: {dups_removed} ({(dups_removed/raw_count*100):.1f}% reduction)")
    report_lines.append(f"   - Missing Values in Cleaned Dataset: {cleaned_df.isnull().sum().sum()}\n")

    # 2. Source Distribution
    report_lines.append("2. SOURCE DISTRIBUTION (Cleaned Dataset)")
    source_counts = cleaned_df['source'].value_counts()
    for src, count in source_counts.items():
        report_lines.append(f"   - {src}: {count} ({count/clean_count*100:.1f}%)")
    report_lines.append("")

    # 3. Category Distribution
    report_lines.append("3. CATEGORY DISTRIBUTION")
    cat_counts = cleaned_df['category'].value_counts()
    for cat, count in cat_counts.items():
        report_lines.append(f"   - {cat:<20}: {count} ({count/clean_count*100:.1f}%)")
    report_lines.append("")

    # 4. Priority Distribution
    report_lines.append("4. PRIORITY DISTRIBUTION")
    pri_counts = cleaned_df['priority'].value_counts()
    for pri, count in pri_counts.items():
        report_lines.append(f"   - {pri:<10}: {count} ({count/clean_count*100:.1f}%)")
    report_lines.append("")

    # 5. Category x Priority Distribution Matrix
    report_lines.append("5. CATEGORY x PRIORITY MATRIX")
    ct_matrix = pd.crosstab(cleaned_df['category'], cleaned_df['priority'])
    report_lines.append(ct_matrix.to_string())
    report_lines.append("")

    # 6. Ticket Text Length Statistics
    word_counts = cleaned_df['ticket_text'].apply(lambda x: len(str(x).split()))
    char_counts = cleaned_df['ticket_text'].apply(lambda x: len(str(x)))
    
    report_lines.append("6. TICKET TEXT LENGTH METRICS")
    report_lines.append(f"   - Word Count -> Mean: {word_counts.mean():.1f}, Min: {word_counts.min()}, Max: {word_counts.max()}, Median: {word_counts.median()}")
    report_lines.append(f"   - Character Count -> Mean: {char_counts.mean():.1f}, Min: {char_counts.min()}, Max: {char_counts.max()}")
    
    shortest_ticket = cleaned_df.loc[word_counts.idxmin()]['ticket_text']
    longest_ticket = cleaned_df.loc[word_counts.idxmax()]['ticket_text']
    report_lines.append(f"   - Shortest Ticket Sample: '{shortest_ticket}'")
    report_lines.append(f"   - Longest Ticket Sample: '{longest_ticket[:100]}...'")
    report_lines.append("")

    # 7. Dataset Splits Breakdown
    report_lines.append("7. SPLITS BREAKDOWN & LEAKAGE PREVENTION")
    report_lines.append(f"   - Training Set (train.csv): {len(train_df)} rows ({len(train_df)/clean_count*100:.1f}%)")
    report_lines.append(f"   - Validation Set (validation.csv): {len(val_df)} rows ({len(val_df)/clean_count*100:.1f}%)")
    report_lines.append(f"   - Test Set (test.csv): {len(test_df)} rows ({len(test_df)/clean_count*100:.1f}%)")
    report_lines.append(f"   - Dedicated Unseen Evaluation Set (unseen_test.csv): {len(unseen_df)} rows")
    report_lines.append(f"   - Dedicated Hinglish Evaluation Set (hinglish_test.csv): {len(hinglish_df)} rows")

    train_texts = set(train_df['ticket_text']) if not train_df.empty else set()
    test_texts = set(test_df['ticket_text']) if not test_df.empty else set()
    overlap = len(train_texts.intersection(test_texts))
    report_lines.append(f"   - Exact Train / Test Text Overlap Count: {overlap} (Target: 0)")
    report_lines.append("\n=========================================================")

    report_text = "\n".join(report_lines)
    report_path = os.path.join(reports_dir, "dataset_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Dataset analysis report generated successfully at {report_path}")
    print("\n--- REPORT SUMMARY ---")
    print(report_text)


if __name__ == "__main__":
    analyze()
