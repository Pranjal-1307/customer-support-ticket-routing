"""
Dataset Validation Script for Customer Support Ticket System.
Validates column structure, data types, value sets, uniqueness, non-emptiness,
and checks for train/validation/test/unseen data leakage.
Exits with code 0 on success, or code 1 with error reports on validation failure.
"""

import os
import sys
import pandas as pd

ALLOWED_CATEGORIES = {
    "Billing",
    "Technical Support",
    "Refund",
    "Shipping",
    "Account",
    "Complaint",
    "Product Inquiry",
    "Cancellation"
}

ALLOWED_PRIORITIES = {"Low", "Medium", "High", "Critical"}
REQUIRED_COLUMNS = ["ticket_id", "ticket_text", "category", "priority", "source"]


def validate_file(file_path: str, allow_duplicates: bool = False) -> tuple[bool, list[str]]:
    errors = []
    if not os.path.exists(file_path):
        errors.append(f"File not found: {file_path}")
        return False, errors

    df = pd.read_csv(file_path)

    # 1. Column presence
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"[{file_path}] Missing required columns: {missing_cols}")

    # 2. Non-empty check
    if len(df) == 0:
        errors.append(f"[{file_path}] Dataset is empty.")
        return False, errors

    # 3. ticket_id uniqueness
    if "ticket_id" in df.columns and not allow_duplicates:
        duplicate_ids = df[df["ticket_id"].duplicated()]["ticket_id"].tolist()
        if duplicate_ids:
            errors.append(f"[{file_path}] Found {len(duplicate_ids)} duplicate ticket_ids.")

    # 4. Null or empty ticket_text
    if "ticket_text" in df.columns:
        null_count = df["ticket_text"].isnull().sum()
        empty_count = (df["ticket_text"].astype(str).str.strip() == "").sum()
        if null_count > 0 or empty_count > 0:
            errors.append(f"[{file_path}] Found {null_count} null and {empty_count} empty ticket_texts.")

    # 5. Category validity
    if "category" in df.columns:
        invalid_cats = set(df["category"].dropna().unique()) - ALLOWED_CATEGORIES
        if invalid_cats:
            errors.append(f"[{file_path}] Found invalid categories: {invalid_cats}")

    # 6. Priority validity
    if "priority" in df.columns:
        invalid_pris = set(df["priority"].dropna().unique()) - ALLOWED_PRIORITIES
        if invalid_pris:
            errors.append(f"[{file_path}] Found invalid priorities: {invalid_pris}")

    # 7. Duplicate ticket_text in same file (enforced only for clean/processed/split sets)
    if "ticket_text" in df.columns and not allow_duplicates:
        dup_texts = df["ticket_text"].duplicated().sum()
        if dup_texts > 0:
            errors.append(f"[{file_path}] Found {dup_texts} duplicate ticket texts.")

    return len(errors) == 0, errors


def validate_leakage(base_dir: str) -> list[str]:
    errors = []
    splits = {
        "train": os.path.join(base_dir, "data", "train.csv"),
        "val": os.path.join(base_dir, "data", "validation.csv"),
        "test": os.path.join(base_dir, "data", "test.csv"),
        "unseen": os.path.join(base_dir, "data", "evaluation", "unseen_test.csv"),
        "hinglish": os.path.join(base_dir, "data", "evaluation", "hinglish_test.csv")
    }

    texts = {}
    for name, path in splits.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "ticket_text" in df.columns:
                texts[name] = set(df["ticket_text"].astype(str).str.strip().str.lower())

    split_names = list(texts.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            name1, name2 = split_names[i], split_names[j]
            overlap = texts[name1].intersection(texts[name2])
            if overlap:
                errors.append(f"Data leakage detected between '{name1}' and '{name2}' ({len(overlap)} overlapping records).")

    return errors


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files_to_validate = [
        (os.path.join(base_dir, "data", "interim", "combined_tickets.csv"), True),
        (os.path.join(base_dir, "data", "processed", "cleaned_tickets.csv"), False),
        (os.path.join(base_dir, "data", "train.csv"), False),
        (os.path.join(base_dir, "data", "validation.csv"), False),
        (os.path.join(base_dir, "data", "test.csv"), False),
        (os.path.join(base_dir, "data", "evaluation", "unseen_test.csv"), False),
        (os.path.join(base_dir, "data", "evaluation", "hinglish_test.csv"), False)
    ]

    all_errors = []
    print("--- Running Dataset Validation Checks ---")
    for fpath, allow_dups in files_to_validate:
        valid, errors = validate_file(fpath, allow_duplicates=allow_dups)
        if valid:
            print(f"PASS: {os.path.basename(fpath)}")
        else:
            print(f"FAIL: {os.path.basename(fpath)}")
            all_errors.extend(errors)

    print("\n--- Running Data Leakage Checks ---")
    leakage_errors = validate_leakage(base_dir)
    if leakage_errors:
        print("FAIL: Leakage checks failed.")
        all_errors.extend(leakage_errors)
    else:
        print("PASS: Zero data leakage across train, val, test, and evaluation sets.")

    if all_errors:
        print("\nValidation Failed with Errors:")
        for err in all_errors:
            print(f" - {err}")
        sys.exit(1)
    else:
        print("\nAll dataset validation checks passed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
