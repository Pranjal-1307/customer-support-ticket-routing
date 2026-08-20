# Dataset Summary Report (Milestone 1 Baseline)

**Generated:** 2026-08-20T06:12:49.114280

**Source file:** `dataset/tickets.csv`

## 1. Basic Statistics

| Metric | Value |
|--------|-------|
| Total records | 1521 |
| Columns | 5 |
| Column names | ticket_id, ticket_text, category, priority, source |
| Text column | ticket_text |
| Category target | category |
| Priority target | priority |
| Department column | N/A |
| Missing values (total) | 0 |

### Missing Values Per Column

| Column | Missing |
|--------|---------|
| ticket_id | 0 |
| ticket_text | 0 |
| category | 0 |
| priority | 0 |
| source | 0 |

### Duplicates

| Type | Count |
|------|-------|
| Exact duplicate rows | 0 |
| Duplicate ticket_text values | 0 |
| Near-duplicate templates (same text after normalizing IDs/amounts/noise) | 1283 |
| Unique templates | 238 |

### Text Length Statistics

| Metric | Words | Characters |
|--------|-------|------------|
| Mean | 14.6 | 91.7 |
| Min | 2 | 14 |
| Max | 21 | 126 |
| Median | 15 | 97 |
| Very short (≤3 words) | 1 | — |
| Very long (≥25 words) | 0 | — |

## 2. Category Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| Billing | 340 | 22.4% |
| Shipping | 278 | 18.3% |
| Refund | 201 | 13.2% |
| Technical Support | 185 | 12.2% |
| Product Inquiry | 166 | 10.9% |
| Cancellation | 155 | 10.2% |
| Account | 100 | 6.6% |
| Complaint | 96 | 6.3% |

## 3. Priority Distribution

| Priority | Count | Percentage |
|----------|-------|------------|
| High | 503 | 33.1% |
| Medium | 498 | 32.7% |
| Critical | 281 | 18.5% |
| Low | 239 | 15.7% |

## 5. Category × Priority Cross-tabulation

| category          |   Critical |   High |   Low |   Medium |
|:------------------|-----------:|-------:|------:|---------:|
| Account           |         21 |     40 |    21 |       18 |
| Billing           |         65 |    133 |    52 |       90 |
| Cancellation      |         10 |     58 |     0 |       87 |
| Complaint         |         23 |     27 |    18 |       28 |
| Product Inquiry   |          0 |      4 |   117 |       45 |
| Refund            |          8 |     63 |    18 |      112 |
| Shipping          |         44 |    132 |    11 |       91 |
| Technical Support |        110 |     46 |     2 |       27 |

## 6. Synthetic Data Observations

- Core dataset is **synthetically generated** using `scripts/generate_data.py`, supplemented by external/manual tickets.
- 10 phrase templates per category × 8 categories = 80 base templates.
- Each template is filled with random IDs/amounts and appended with 1 of 8 noise suffixes.
- **1283** records are near-duplicates (same template, different random fill).
- Category-specific keywords (e.g., 'refund', 'billing', 'cancel') appear **exclusively** in their respective category templates.
- Priority is **deterministically derived** from template base-priority + urgency suffix upgrade.
- Urgency suffixes ('Urgent attention required.', 'Please advise as soon as possible.') upgrade base priority by 1 level.
- This makes category classification trivially separable and priority partially learnable from text.