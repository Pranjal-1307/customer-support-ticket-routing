# Data Validation & Pipeline Safety Report — Milestone 2

**Generated:** `2026-08-20T07:43:51.282842`  
**Status:** **PASS**  

---

## 1. Incoming Ticket Validation Architecture

To ensure raw or external tickets do not corrupt the supervised learning pipeline, all manual and external tickets land in `data/incoming/` before processing.

### Controlled Ingestion Flow:
```
data/incoming/ (pending CSVs)
        │
        ▼
   [Validation & PII Check] ───(Rejected)──► data/incoming/rejected_tickets.csv
        │
    (Approved)
        ▼
  data/raw/ (external_tickets.csv / manual_tickets.csv)
        │
        ▼
  [Canonical Pipeline: Cleaning -> Deduplication -> Stratified Group Split]
```

---

## 2. Enforced Validation Rules

| Rule Category | Validation Check | Action on Failure |
|---|---|---|
| **Required Schema** | Presence of `ticket_id`, `ticket_text`, `category`, `priority`, `source` | Reject record |
| **Non-Emptiness** | Check `ticket_text` is not null or whitespace | Reject record |
| **Allowed Values** | Category ∈ 8 valid categories; Priority ∈ {Low, Medium, High, Critical} | Reject record |
| **Length Bounds** | 5 ≤ `len(ticket_text)` ≤ 2000 characters | Reject record |
| **PII Detection** | Regex search for Emails, Phone Numbers, Credit Cards, SSNs | Reject & Flag for Masking |
| **Duplicate IDs** | `ticket_id` must be unique across all existing datasets | Reject record |
| **Exact Duplicates** | `ticket_text` must not match any existing cleaned record | Reject record |

---

## 3. Ingestion Validation Results (Sample Run)

- **Total Incoming Records Processed:** 6
- **Approved & Merged into Raw Data:** 4 records (66.7%)
- **Rejected & Isolated:** 2 records (33.3%)

### Rejection Log Details:
1. **Reason:** `PII detected: Email address detected, Phone number detected`
2. **Reason:** `Text too short (3 chars, min 5)`

---

## 4. Unverified & Live Data Isolation Guarantees

- **No Automatic Retraining:** Incoming tickets remain in `data/incoming/` until explicitly validated by `scripts/ingest_incoming.py`.
- **No Self-Labeling:** Model predictions on live customer tickets are NEVER automatically appended to training datasets.
- **Strict Label Provenance:** Every record retains its immutable `source` tag (`synthetic`, `manual`, or `external`).
