# Phase 9–12: Human Feedback & Controlled Retraining Workflow

## 1. Architectural Philosophy: Human-in-the-Loop Oversight
In real-world customer support routing, automatic self-learning from raw uncurated inputs can lead to catastrophic forgetting, feedback loops, and vulnerability to adversarial manipulation. 

M3 establishes a **strictly controlled, validated feedback architecture**:

```
Customer Ticket Prediction (Production Model)
       ↓
Prediction Display (Priority + Confidence + Interpretable Signals)
       ↓
Human / Agent Priority Correction (Admin UI / API)
       ↓
Isolated SQLite Feedback Store (priority_feedback table, validated=0)
       ↓
Supervisor Quality Gate (Review & Validation: validated=1)
       ↓
Controlled Candidate Retraining Pipeline (scripts/retrain_with_feedback.py)
       ↓
Holdout Evaluation Gate vs Production Model
       ↓
Explicit Staging & Promotion (models/model_version.json)
```

---

## 2. Feedback Schema & Storage
Feedback is stored separately in SQLite (`tickets.db`) without modifying the original ticket prediction record:

```sql
CREATE TABLE IF NOT EXISTS priority_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feedback_id TEXT UNIQUE NOT NULL,
    ticket_id TEXT NOT NULL,
    ticket_text TEXT,
    original_priority TEXT NOT NULL,
    corrected_priority TEXT NOT NULL,
    original_priority_confidence REAL,
    reason TEXT,
    reviewer TEXT DEFAULT 'agent',
    validated INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(ticket_id) REFERENCES tickets(ticket_id)
);
```

---

## 3. REST API Interface

### A. Submit Priority Correction
- **Endpoint:** `POST /api/feedback/priority`
- **Payload:**
```json
{
    "ticket_id": "TICK-A1B2C3",
    "corrected_priority": "Medium",
    "reason": "Routine status inquiry; no operational defect.",
    "reviewer": "supervisor_alice"
}
```
- **Response (201 Created):**
```json
{
    "status": "success",
    "feedback_id": "FB-89E2FA",
    "ticket_id": "TICK-A1B2C3",
    "original_priority": "High",
    "corrected_priority": "Medium",
    "validated": 0
}
```

### B. Fetch All Feedback Records
- **Endpoint:** `GET /api/feedback/priority`
- **Response (200 OK):** Returns list of all submitted corrections with reviewer notes and validation state.

### C. Validate Feedback for Retraining Gate
- **Endpoint:** `POST /api/feedback/priority/validate`
- **Payload:**
```json
{
    "feedback_id": "FB-89E2FA",
    "validated": 1
}
```

---

## 4. Controlled Retraining Script
- **Script:** `scripts/retrain_with_feedback.py`
- **Execution:**
```bash
python scripts/retrain_with_feedback.py
```
- **With Explicit Promotion:**
```bash
python scripts/retrain_with_feedback.py --promote
```

### Validation Gate Criteria:
1. **Holdout Quality Threshold:** Candidate holdout accuracy must be greater than or equal to current active production holdout accuracy.
2. **Validation Integrity:** Candidate cross-validation accuracy must maintain $\ge 85.0\%$.
3. **Audit Trail:** Model version metadata is recorded in `models/model_version.json` and artifact is saved to `models/priority_classifier_v2.pkl`.
