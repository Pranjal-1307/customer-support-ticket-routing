# Dataset Versioning Report — Milestone 2 (v2.0-M2)

**Version:** `v2.0-M2`  
**Build Timestamp:** `2026-08-20T11:06:19.922582`  

---

## 1. Summary Statistics

| Metric | Count |
|---|---|
| Total Combined Raw Records | 5761 |
| Cleaned Processed Records | 4341 |
| Unique Ticket Texts | 4341 |
| Training Split (70%) | 3103 |
| Validation Split (15%) | 636 |
| Test Split (15%) | 602 |

## 2. Source Breakdown

| Source | Count | Percentage |
|---|---|---|
| `synthetic` | 4180 | 96.3% |
| `manual` | 84 | 1.9% |
| `external` | 77 | 1.8% |

## 3. Category Distribution

| Category | Count | Percentage |
|---|---|---|
| Billing | 570 | 13.1% |
| Refund | 566 | 13.0% |
| Technical Support | 558 | 12.9% |
| Shipping | 542 | 12.5% |
| Account | 538 | 12.4% |
| Product Inquiry | 528 | 12.2% |
| Cancellation | 527 | 12.1% |
| Complaint | 512 | 11.8% |

## 4. Priority Distribution

| Priority | Count | Percentage |
|---|---|---|
| Medium | 1682 | 38.7% |
| High | 1230 | 28.3% |
| Low | 893 | 20.6% |
| Critical | 536 | 12.3% |

## 5. Leakage Verification

- **Exact Text Overlap (Train vs Test):** 0 records
- **Template Pattern Overlap (Train vs Test):** 0 groups
- **Normalization Method:** Entity masking & noise phrase removal
