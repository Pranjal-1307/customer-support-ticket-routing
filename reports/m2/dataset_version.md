# Dataset Versioning Report — Milestone 2 (v2.0-M2)

**Version:** `v2.0-M2`  
**Build Timestamp:** `2026-09-26T20:13:13.749111`  

---

## 1. Summary Statistics

| Metric | Count |
|---|---|
| Total Combined Raw Records | 5761 |
| Cleaned Processed Records | 3850 |
| Unique Ticket Texts | 3850 |
| Training Split (70%) | 2648 |
| Validation Split (15%) | 639 |
| Test Split (15%) | 563 |

## 2. Source Breakdown

| Source | Count | Percentage |
|---|---|---|
| `synthetic` | 3689 | 95.8% |
| `manual` | 84 | 2.2% |
| `external` | 77 | 2.0% |

## 3. Category Distribution

| Category | Count | Percentage |
|---|---|---|
| Billing | 533 | 13.8% |
| Shipping | 514 | 13.4% |
| Technical Support | 495 | 12.9% |
| Refund | 489 | 12.7% |
| Cancellation | 469 | 12.2% |
| Product Inquiry | 451 | 11.7% |
| Account | 451 | 11.7% |
| Complaint | 448 | 11.6% |

## 4. Priority Distribution

| Priority | Count | Percentage |
|---|---|---|
| Medium | 1547 | 40.2% |
| High | 985 | 25.6% |
| Low | 939 | 24.4% |
| Critical | 379 | 9.8% |

## 5. Leakage Verification

- **Exact Text Overlap (Train vs Test):** 0 records
- **Template Pattern Overlap (Train vs Test):** 0 groups
- **Normalization Method:** Entity masking & noise phrase removal
