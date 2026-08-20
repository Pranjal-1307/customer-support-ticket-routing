"""
Incoming Ticket Ingestion and Validation Pipeline for Customer Support Ticket System.
Processes incoming external/manual tickets from data/incoming/, validates schema, checks for
PII, duplicate IDs/texts, and text length bounds. Sets validation_status to 'approved' or 'rejected'.
Appends approved tickets to data/raw/ safely while isolating rejected/unverified tickets.
"""

import os
import sys
import re
import datetime
import pandas as pd

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

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
ALLOWED_SOURCES = {"synthetic", "manual", "external"}

# PII Regex Patterns
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
PHONE_REGEX = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
CREDIT_CARD_REGEX = re.compile(r'\b(?:\d[ -]*?){13,16}\b')
SSN_REGEX = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')


def check_pii(text: str) -> list[str]:
    """Detects potential PII (emails, phone numbers, credit card numbers, SSNs) in text."""
    pii_found = []
    if EMAIL_REGEX.search(text):
        pii_found.append("Email address detected")
    if PHONE_REGEX.search(text):
        pii_found.append("Phone number detected")
    if CREDIT_CARD_REGEX.search(text):
        pii_found.append("Credit card sequence detected")
    if SSN_REGEX.search(text):
        pii_found.append("SSN pattern detected")
    return pii_found


def validate_incoming_record(row: pd.Series, existing_ids: set, existing_texts: set) -> tuple[bool, list[str]]:
    """Validates a single incoming ticket record against business rules."""
    reasons = []

    # 1. Missing required fields
    ticket_id = str(row.get("ticket_id", "")).strip()
    ticket_text = str(row.get("ticket_text", "")).strip()
    category = str(row.get("category", "")).strip().title()
    priority = str(row.get("priority", "")).strip().title()
    source = str(row.get("source", "external")).strip().lower()

    if not ticket_id or ticket_id == "nan":
        reasons.append("Missing ticket_id")
    elif ticket_id in existing_ids:
        reasons.append(f"Duplicate ticket_id '{ticket_id}'")

    if not ticket_text or ticket_text == "nan":
        reasons.append("Missing or empty ticket_text")
    else:
        # Length check
        if len(ticket_text) < 5:
            reasons.append(f"Text too short ({len(ticket_text)} chars, min 5)")
        elif len(ticket_text) > 2000:
            reasons.append(f"Text too long ({len(ticket_text)} chars, max 2000)")

        # Duplicate text check
        if ticket_text.lower() in existing_texts:
            reasons.append("Exact duplicate ticket_text already in dataset")

        # PII check
        pii_flags = check_pii(ticket_text)
        if pii_flags:
            reasons.append(f"PII detected: {', '.join(pii_flags)}")

    # Category validity
    if not category or category not in ALLOWED_CATEGORIES:
        reasons.append(f"Invalid category '{row.get('category')}'")

    # Priority validity
    if not priority or priority not in ALLOWED_PRIORITIES:
        reasons.append(f"Invalid priority '{row.get('priority')}'")

    # Source validity
    if source not in ALLOWED_SOURCES:
        reasons.append(f"Invalid source '{row.get('source')}'")

    is_valid = len(reasons) == 0
    return is_valid, reasons


def process_incoming_tickets():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    incoming_dir = os.path.join(base_dir, "data", "incoming")
    raw_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(incoming_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    # Gather existing IDs and texts across data/raw/ files to prevent duplicates
    existing_ids = set()
    existing_texts = set()

    for fname in ["synthetic_tickets.csv", "external_tickets.csv", "manual_tickets.csv"]:
        fpath = os.path.join(raw_dir, fname)
        if os.path.exists(fpath):
            df_ex = pd.read_csv(fpath)
            if "ticket_id" in df_ex.columns:
                existing_ids.update(df_ex["ticket_id"].astype(str).str.strip().tolist())
            if "ticket_text" in df_ex.columns:
                existing_texts.update(df_ex["ticket_text"].astype(str).str.strip().str.lower().tolist())

    incoming_files = [f for f in os.listdir(incoming_dir) if f.endswith(".csv") and not f.startswith("processed_") and not f.startswith("rejected_")]

    accepted_records = []
    rejected_records = []

    print("--- STEP: Processing Incoming Tickets ---")
    if not incoming_files:
        print("No pending incoming CSV files found in data/incoming/.")
        return 0, 0

    for fname in incoming_files:
        fpath = os.path.join(incoming_dir, fname)
        print(f"Reading incoming file: {fname}")
        df_in = pd.read_csv(fpath)

        for idx, row in df_in.iterrows():
            is_valid, reasons = validate_incoming_record(row, existing_ids, existing_texts)
            
            rec_dict = row.to_dict()
            rec_dict["source"] = str(row.get("source", "external")).strip().lower()
            rec_dict["created_at"] = row.get("created_at", datetime.datetime.now().isoformat())

            if is_valid:
                rec_dict["validation_status"] = "approved"
                rec_dict["rejection_reason"] = ""
                accepted_records.append(rec_dict)
                existing_ids.add(str(rec_dict["ticket_id"]))
                existing_texts.add(str(rec_dict["ticket_text"]).strip().lower())
            else:
                rec_dict["validation_status"] = "rejected"
                rec_dict["rejection_reason"] = "; ".join(reasons)
                rejected_records.append(rec_dict)

    # Save validation reports in data/incoming/
    if accepted_records:
        df_acc = pd.DataFrame(accepted_records)
        df_acc_cols = ["ticket_id", "ticket_text", "category", "priority", "source", "created_at", "validation_status"]
        df_acc = df_acc[[c for c in df_acc_cols if c in df_acc.columns]]

        # Append to data/raw/external_tickets.csv or data/raw/manual_tickets.csv based on source
        external_path = os.path.join(raw_dir, "external_tickets.csv")
        manual_path = os.path.join(raw_dir, "manual_tickets.csv")

        manual_acc = df_acc[df_acc["source"] == "manual"]
        ext_acc = df_acc[df_acc["source"] != "manual"]

        if not manual_acc.empty:
            if os.path.exists(manual_path):
                old_man = pd.read_csv(manual_path)
                combined_man = pd.concat([old_man, manual_acc], ignore_index=True).drop_duplicates(subset=["ticket_text"])
                combined_man.to_csv(manual_path, index=False)
            else:
                manual_acc.to_csv(manual_path, index=False)

        if not ext_acc.empty:
            if os.path.exists(external_path):
                old_ext = pd.read_csv(external_path)
                combined_ext = pd.concat([old_ext, ext_acc], ignore_index=True).drop_duplicates(subset=["ticket_text"])
                combined_ext.to_csv(external_path, index=False)
            else:
                ext_acc.to_csv(external_path, index=False)

        print(f"Approved {len(accepted_records)} incoming tickets and merged into raw datasets.")

    if rejected_records:
        df_rej = pd.DataFrame(rejected_records)
        rej_log_path = os.path.join(incoming_dir, "rejected_tickets.csv")
        df_rej.to_csv(rej_log_path, index=False)
        print(f"Rejected {len(rejected_records)} tickets logged to {rej_log_path}.")

    print(f"Ingestion complete: Accepted={len(accepted_records)}, Rejected={len(rejected_records)}")
    return len(accepted_records), len(rejected_records)


if __name__ == "__main__":
    process_incoming_tickets()
