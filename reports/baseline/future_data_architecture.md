# Future Data Architecture (Design Document)

**Generated:** 2026-08-20T06:13:18.585044

**Status:** Design only — NOT implemented in Milestone 1

## Current State (M1)

```
Synthetic Dataset → 80/20 Split → TF-IDF → Train → Deploy → Predict
                                                          ↓
                                                   /api/admin/retrain
                                                   (overwrites model)
```

## Target Architecture (Future Milestones)

```
┌─────────────────────────────────────────────────────────────────┐
│ DATA SOURCES                                                    │
│  1. Existing synthetic dataset (initial training data)          │
│  2. Realistic/manual customer tickets                           │
│  3. Live incoming tickets (predictions only, NOT auto-labeled)  │
│  4. Human/admin corrections of incorrect predictions            │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ FEEDBACK & VALIDATION PIPELINE                                  │
│  5. Validated corrected labels (admin-approved)                 │
│  6. Versioned feedback dataset (append-only, timestamped)       │
│     - Each record: ticket_text, predicted_label,                │
│       corrected_label, reviewer, timestamp, source              │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ CONTROLLED RETRAINING                                           │
│  7. Merge synthetic + validated feedback data                   │
│  8. Evaluate against fixed validation/test set                  │
│  9. Compare new model vs current production model               │
│ 10. Require admin approval before replacing production model    │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ DEPLOYMENT                                                      │
│  - Model versioning (v1, v2, ...)                              │
│  - Rollback capability                                          │
│  - A/B testing (optional future)                               │
└─────────────────────────────────────────────────────────────────┘
```

## Critical Design Principle

> **A live ticket should NOT automatically become trusted training data
> merely because the model predicted a label.**

> The model's own prediction must be reviewed and corrected by a human
> (admin/agent) before it enters the training pipeline.

## Feedback Loop Flow

```
New Ticket → Model Prediction → Display to Agent/Admin
                                        ↓
                              Correct? ─── Yes → Archive (no action)
                                 │
                                 No
                                 ↓
                          Admin Corrects Label
                                 ↓
                      Correction Validated & Stored
                                 ↓
                      Added to Feedback Dataset (versioned)
                                 ↓
                    [Periodic / Triggered Retraining]
                                 ↓
                    Evaluate New Model vs Current
                                 ↓
                    Admin Approves → Deploy New Model
```

## Implementation Notes for Later Milestones

- Store feedback in a separate table/CSV with provenance metadata.
- Never overwrite the original synthetic training set.
- Maintain a held-out test set that is never modified or trained on.
- Log model performance before and after each retraining cycle.
- Add model version metadata to predictions for traceability.