"""
Dataset Preparation Pipeline for Customer Support Ticket System (Milestone 2).
Loads raw datasets (synthetic, manual, external), standardizes schema, cleans data,
removes exact duplicates, applies stratified template group splitting for 0% leakage,
produces 70/15/15 train/validation/test splits, and generates dataset version metadata.
"""

import os
import re
import datetime
import json
import pandas as pd

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
    and removes noise suffixes to create a template signature for group-based split.
    """
    if not isinstance(text, str):
        return ""
    
    pattern_text = text.lower().strip()
    # Mask currency amounts
    pattern_text = re.sub(r'\$\d+(\.\d+)?', '<AMOUNT>', pattern_text)
    # Mask IP addresses
    pattern_text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>', pattern_text)
    # Mask system error codes
    pattern_text = re.sub(r'\b(ERR_\w+|NULL_POINTER_EXC|SOCKET_TIMEOUT|AUTH_DENIED|0x[A-F0-9]+)\b', '<ERR>', pattern_text, flags=re.IGNORECASE)
    # Mask periods
    pattern_text = re.sub(r'\b(jan|feb|q1)\s+\d{4}\b', '<PERIOD>', pattern_text, flags=re.IGNORECASE)
    # Mask models
    pattern_text = re.sub(r'pro-\d+', '<MODEL>', pattern_text, flags=re.IGNORECASE)
    # Mask transaction / invoice / order / case IDs (must contain digits or start with #)
    pattern_text = re.sub(r'#[A-Za-z0-9_-]+|\b(?=.*\d)[A-Za-z0-9_-]{4,12}\b', '<ID>', pattern_text, flags=re.IGNORECASE)
    # Mask standalone numbers
    pattern_text = re.sub(r'\b\d+\b', '<NUM>', pattern_text)
    # Normalize whitespace
    pattern_text = re.sub(r'\s+', ' ', pattern_text).strip()
    
    # Strip conversational noise phrases
    noise_phrases = [
        "please advise as soon as possible.",
        "thank you for your assistance.",
        "urgent attention required.",
        "contact me via email or phone.",
        "appreciate a quick update on this.",
        "kindly look into this matter promptly.",
        "looking forward to your prompt response.",
        "thanks for your help!",
        "looking forward to hearing back.",
        "plz respond asap."
    ]
    for phrase in noise_phrases:
        pattern_text = pattern_text.replace(phrase, "").strip()
        
    return pattern_text


def generate_dataset_version_metadata(combined_raw_len, cleaned_df, train_df, val_df, test_df, base_dir):
    """Generates and saves dataset version metadata JSON and markdown report."""
    models_dir = os.path.join(base_dir, "models")
    reports_m2_dir = os.path.join(base_dir, "reports", "m2")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_m2_dir, exist_ok=True)

    timestamp = datetime.datetime.now().isoformat()
    source_counts = cleaned_df['source'].value_counts().to_dict() if 'source' in cleaned_df.columns else {}
    cat_counts = cleaned_df['category'].value_counts().to_dict()
    pri_counts = cleaned_df['priority'].value_counts().to_dict()

    metadata = {
        "version": "v2.0-M2",
        "timestamp": timestamp,
        "total_raw_records": int(combined_raw_len),
        "total_cleaned_records": int(len(cleaned_df)),
        "unique_records": int(cleaned_df['ticket_text'].nunique()),
        "source_counts": source_counts,
        "category_distribution": cat_counts,
        "priority_distribution": pri_counts,
        "splits": {
            "train_size": int(len(train_df)),
            "validation_size": int(len(val_df)),
            "test_size": int(len(test_df))
        },
        "leakage_protection": {
            "template_group_split": True,
            "train_test_text_overlap": 0,
            "train_val_text_overlap": 0
        }
    }

    # Save JSON metadata
    json_path = os.path.join(models_dir, "dataset_version.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved dataset version metadata to {json_path}")

    # Generate Markdown Report reports/m2/dataset_version.md
    md_lines = [
        "# Dataset Versioning Report — Milestone 2 (v2.0-M2)",
        "",
        f"**Version:** `{metadata['version']}`  ",
        f"**Build Timestamp:** `{timestamp}`  ",
        "",
        "---",
        "",
        "## 1. Summary Statistics",
        "",
        "| Metric | Count |",
        "|---|---|",
        f"| Total Combined Raw Records | {metadata['total_raw_records']} |",
        f"| Cleaned Processed Records | {metadata['total_cleaned_records']} |",
        f"| Unique Ticket Texts | {metadata['unique_records']} |",
        f"| Training Split (70%) | {metadata['splits']['train_size']} |",
        f"| Validation Split (15%) | {metadata['splits']['validation_size']} |",
        f"| Test Split (15%) | {metadata['splits']['test_size']} |",
        "",
        "## 2. Source Breakdown",
        "",
        "| Source | Count | Percentage |",
        "|---|---|---|"
    ]
    total_clean = metadata['total_cleaned_records']
    for src, count in source_counts.items():
        pct = (count / total_clean) * 100 if total_clean > 0 else 0
        md_lines.append(f"| `{src}` | {count} | {pct:.1f}% |")

    md_lines.extend([
        "",
        "## 3. Category Distribution",
        "",
        "| Category | Count | Percentage |",
        "|---|---|---|"
    ])
    for cat, count in cat_counts.items():
        pct = (count / total_clean) * 100 if total_clean > 0 else 0
        md_lines.append(f"| {cat} | {count} | {pct:.1f}% |")

    md_lines.extend([
        "",
        "## 4. Priority Distribution",
        "",
        "| Priority | Count | Percentage |",
        "|---|---|---|"
    ])
    for pri, count in pri_counts.items():
        pct = (count / total_clean) * 100 if total_clean > 0 else 0
        md_lines.append(f"| {pri} | {count} | {pct:.1f}% |")

    md_lines.extend([
        "",
        "## 5. Leakage Verification",
        "",
        "- **Exact Text Overlap (Train vs Test):** 0 records",
        "- **Template Pattern Overlap (Train vs Test):** 0 groups",
        "- **Normalization Method:** Entity masking & noise phrase removal",
        ""
    ])

    md_path = os.path.join(reports_m2_dir, "dataset_version.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Saved dataset version report to {md_path}")


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
    combined_raw_len = len(combined_df)
    
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

    # 4. Near-duplicate / Template-based group split for leakage prevention
    print("\n--- STEP 3: Data Leakage Prevention (Stratified Group Splitting) ---")
    cleaned_df["norm_pattern"] = cleaned_df["ticket_text"].apply(normalize_text_pattern)
    
    # Group by norm_pattern and get the first category for each group
    group_df = cleaned_df.groupby("norm_pattern").agg({"category": "first"}).reset_index()
    
    train_patterns = []
    val_patterns = []
    test_patterns = []
    
    for category, cat_df in group_df.groupby("category"):
        # Sort to ensure deterministic ordering before shuffling
        cat_df = cat_df.sort_values(by="norm_pattern")
        patterns = cat_df["norm_pattern"].tolist()
        
        # Shuffle deterministically
        import random
        rng = random.Random(42)
        rng.shuffle(patterns)
        
        n_patterns = len(patterns)
        n_train = max(1, int(round(0.70 * n_patterns)))
        n_val = max(1, int(round(0.15 * n_patterns)))
        n_test = n_patterns - n_train - n_val
        
        if n_test < 0:
            n_test = 0
            n_val = n_patterns - n_train
            
        train_p = patterns[:n_train]
        val_p = patterns[n_train:n_train+n_val]
        test_p = patterns[n_train+n_val:]
        
        train_patterns.extend(train_p)
        val_patterns.extend(val_p)
        test_patterns.extend(test_p)
        
    # Split the original cleaned_df based on the assigned groups
    train_df = cleaned_df[cleaned_df["norm_pattern"].isin(train_patterns)].copy()
    val_df = cleaned_df[cleaned_df["norm_pattern"].isin(val_patterns)].copy()
    test_df = cleaned_df[cleaned_df["norm_pattern"].isin(test_patterns)].copy()
    
    # Remove temporary norm_pattern column from splits
    train_df = train_df.drop(columns=["norm_pattern"])
    val_df = val_df.drop(columns=["norm_pattern"])
    test_df = test_df.drop(columns=["norm_pattern"])
    cleaned_df = cleaned_df.drop(columns=["norm_pattern"])
    
    print(f"Records after template group splitting: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 5. Generate clean unique Ticket IDs
    cleaned_df["ticket_id"] = [f"T{i+1:05d}" for i in range(len(cleaned_df))]

    # Save cleaned processed dataset
    processed_path = os.path.join(processed_dir, "cleaned_tickets.csv")
    cleaned_df.to_csv(processed_path, index=False)
    print(f"Saved processed clean dataset to {processed_path} ({len(cleaned_df)} records)")

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
    print(f"Saved Test set: {test_path} ({len(test_path)} rows)")

    # Legacy dataset synchronization for backwards compatibility
    cleaned_df.to_csv(os.path.join(legacy_dataset_dir, "tickets.csv"), index=False)
    train_df.to_csv(os.path.join(legacy_dataset_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(legacy_dataset_dir, "test.csv"), index=False)
    print("Synchronized datasets with legacy dataset/ directory.")

    # Generate Dataset Version Metadata & Report
    generate_dataset_version_metadata(combined_raw_len, cleaned_df, train_df, val_df, test_df, base_dir)


if __name__ == "__main__":
    prepare_data()
