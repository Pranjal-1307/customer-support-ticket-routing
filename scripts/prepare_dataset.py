"""
Dataset Preparation Pipeline for Customer Support Ticket System.
Loads raw datasets, standardizes schema, cleans data, removes exact & template duplicates,
prevents data leakage, and produces stratified 70/15/15 train/validation/test splits.
"""

import os
import re
import random
import pandas as pd
from sklearn.model_selection import train_test_split

ALLOWED_CATEGORIES = [
    "Billing",
    "Technical Support",
    "Refund",
    "Shipping",
    "Account",
    "Complaint",
    "Product Inquiry",
    "Cancellation"
]

ALLOWED_PRIORITIES = ["Low", "Medium", "High", "Critical"]


def normalize_text_pattern(text: str) -> str:
    """
    Normalizes dynamic entities (amounts, IDs, hex codes, IP addresses, numbers)
    in ticket text to create a template signature for near-duplicate detection.
    """
    if not isinstance(text, str):
        return ""
    
    pattern_text = text.lower().strip()
    # Mask currency amounts
    pattern_text = re.sub(r'\$\d+(\.\d+)?', '<AMOUNT>', pattern_text)
    # Mask transaction / invoice / order / case IDs
    pattern_text = re.sub(r'#?[A-Z0-9]{4,12}\b', '<ID>', pattern_text, flags=re.IGNORECASE)
    # Mask IP addresses
    pattern_text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>', pattern_text)
    # Mask standalone numbers
    pattern_text = re.sub(r'\b\d+\b', '<NUM>', pattern_text)
    # Normalize whitespace
    pattern_text = re.sub(r'\s+', ' ', pattern_text)
    return pattern_text


def prepare_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")
    interim_dir = os.path.join(base_dir, "data", "interim")
    processed_dir = os.path.join(base_dir, "data", "processed")
    data_dir = os.path.join(base_dir, "data")
    legacy_dataset_dir = os.path.join(base_dir, "dataset")

    os.makedirs(interim_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(legacy_dataset_dir, exist_ok=True)

    raw_files = [
        ("synthetic_tickets.csv", "synthetic"),
        ("external_tickets.csv", "external"),
        ("manual_tickets.csv", "manual")
    ]

    dfs = []
    print("--- STEP 1: Loading Raw Datasets ---")
    for fname, default_source in raw_files:
        fpath = os.path.join(raw_dir, fname)
        if os.path.exists(fpath):
            df_temp = pd.read_csv(fpath)
            if 'source' not in df_temp.columns:
                df_temp['source'] = default_source
            dfs.append(df_temp)
            print(f"Loaded {fname}: {len(df_temp)} records")
        else:
            print(f"Warning: Raw file {fname} not found.")

    if not dfs:
        raise FileNotFoundError("No raw dataset files found in data/raw/")

    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Standardize column names
    required_cols = ["ticket_id", "ticket_text", "category", "priority", "source"]
    for col in required_cols:
        if col not in combined_df.columns:
            raise KeyError(f"Required column '{col}' missing from raw data.")

    combined_df = combined_df[required_cols]
    
    # Save combined interim dataset
    interim_path = os.path.join(interim_dir, "combined_tickets.csv")
    combined_df.to_csv(interim_path, index=False)
    print(f"\nSaved combined interim dataset to {interim_path} ({len(combined_df)} records)")

    print("\n--- STEP 2: Cleaning and Normalizing ---")
    # 1. Remove empty/null ticket_text, category, priority
    cleaned_df = combined_df.dropna(subset=["ticket_text", "category", "priority"]).copy()
    cleaned_df["ticket_text"] = cleaned_df["ticket_text"].astype(str).str.strip()
    cleaned_df = cleaned_df[cleaned_df["ticket_text"] != ""]

    # 2. Normalize categories and priorities
    cleaned_df["category"] = cleaned_df["category"].str.strip().str.title()
    cleaned_df["priority"] = cleaned_df["priority"].str.strip().str.title()

    # Filter invalid categories or priorities
    cleaned_df = cleaned_df[cleaned_df["category"].isin(ALLOWED_CATEGORIES)]
    cleaned_df = cleaned_df[cleaned_df["priority"].isin(ALLOWED_PRIORITIES)]

    print(f"Records after null & invalid filtering: {len(cleaned_df)}")

    # 3. Exact text deduplication
    before_exact_dedup = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates(subset=["ticket_text"]).copy()
    exact_dups_removed = before_exact_dedup - len(cleaned_df)
    print(f"Exact duplicates removed: {exact_dups_removed}")

    # 4. Near-duplicate / Template-based deduplication for leakage prevention
    print("\n--- STEP 3: Data Leakage Prevention (Template Deduplication) ---")
    cleaned_df["norm_pattern"] = cleaned_df["ticket_text"].apply(normalize_text_pattern)
    
    # Prioritize 'manual' and 'external' sources over 'synthetic' during template deduplication
    source_priority_map = {"manual": 0, "external": 1, "synthetic": 2}
    cleaned_df["src_rank"] = cleaned_df["source"].map(lambda s: source_priority_map.get(str(s).lower(), 3))
    
    cleaned_df = cleaned_df.sort_values(by="src_rank").drop_duplicates(subset=["norm_pattern"]).copy()
    cleaned_df = cleaned_df.drop(columns=["norm_pattern", "src_rank"])
    
    print(f"Records after template pattern deduplication: {len(cleaned_df)}")

    # 5. Generate clean unique Ticket IDs
    cleaned_df["ticket_id"] = [f"T{i+1:05d}" for i in range(len(cleaned_df))]

    # Save cleaned processed dataset
    processed_path = os.path.join(processed_dir, "cleaned_tickets.csv")
    cleaned_df.to_csv(processed_path, index=False)
    print(f"Saved processed clean dataset to {processed_path} ({len(cleaned_df)} records)")

    print("\n--- STEP 4: Stratified Train / Validation / Test Splitting ---")
    # Split 70% Train, 15% Validation, 15% Test (or 70% Train, 30% Temp -> 15% Val, 15% Test)
    train_df, temp_df = train_test_split(
        cleaned_df,
        test_size=0.30,
        random_state=42,
        stratify=cleaned_df["category"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["category"]
    )

    # Verify no overlap between splits
    train_texts = set(train_df["ticket_text"])
    val_texts = set(val_df["ticket_text"])
    test_texts = set(test_df["ticket_text"])

    overlap_train_val = len(train_texts.intersection(val_texts))
    overlap_train_test = len(train_texts.intersection(test_texts))
    overlap_val_test = len(val_texts.intersection(test_texts))

    print(f"Train/Val Overlap: {overlap_train_val}")
    print(f"Train/Test Overlap: {overlap_train_test}")
    print(f"Val/Test Overlap: {overlap_val_test}")

    # Save data splits
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "validation.csv")
    test_path = os.path.join(data_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nSaved Train set: {train_path} ({len(train_df)} rows)")
    print(f"Saved Validation set: {val_path} ({len(val_df)} rows)")
    print(f"Saved Test set: {test_path} ({len(test_df)} rows)")

    # Legacy dataset synchronization for backwards compatibility
    cleaned_df.to_csv(os.path.join(legacy_dataset_dir, "tickets.csv"), index=False)
    train_df.to_csv(os.path.join(legacy_dataset_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(legacy_dataset_dir, "test.csv"), index=False)
    print("Synchronized datasets with legacy dataset/ directory.")


if __name__ == "__main__":
    prepare_data()
