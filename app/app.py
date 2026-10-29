"""
Flask Application for Customer Support Ticket Routing System using NLP.
Handles customer ticket submission, sqlite persistence, NLP classification & routing,
admin analytics endpoints, and model retraining.
"""

import os
import sqlite3
import uuid
import datetime
from flask import Flask, render_template, request, jsonify

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.predict import get_predictor, predict_ticket
from utils.ticket_validation import validate_ticket_text
from scripts.train_models import train_and_evaluate_models

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "tickets.db")
DEMO_AGENTS = [
    ("Demo Agent A1", "Billing Team"), ("Demo Agent A2", "Billing Team"),
    ("Demo Agent B1", "Technical Support Team"), ("Demo Agent B2", "Technical Support Team"),
    ("Demo Agent C1", "Refund Team"), ("Demo Agent C2", "Refund Team"),
    ("Demo Agent D1", "Logistics Team"), ("Demo Agent D2", "Logistics Team"),
    ("Demo Agent E1", "Account Support Team"), ("Demo Agent E2", "Account Support Team"),
    ("Demo Agent F1", "Customer Care Team"), ("Demo Agent F2", "Customer Care Team"),
    ("Demo Agent G1", "Product Support Team"), ("Demo Agent G2", "Product Support Team"),
    ("Demo Agent H1", "Retention Team"), ("Demo Agent H2", "Retention Team"),
    ("Demo Agent I1", "General Support Team"), ("Demo Agent I2", "General Support Team"),
]


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Initializes SQLite database tables and populates initial sample data if empty."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1,
            UNIQUE(name, department)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id TEXT UNIQUE NOT NULL,
            ticket_text TEXT NOT NULL,
            category TEXT NOT NULL,
            priority TEXT NOT NULL,
            department TEXT NOT NULL,
            confidence REAL NOT NULL,
            category_confidence REAL,
            priority_confidence REAL,
            status TEXT DEFAULT 'Open',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            assigned_agent_id INTEGER REFERENCES agents(id) ON DELETE SET NULL
        )
    """)
    cursor.execute("""
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    # Automatic schema migration for existing databases
    cursor.execute("PRAGMA table_info(tickets)")
    existing_cols = [row[1] for row in cursor.fetchall()]
    if "category_confidence" not in existing_cols:
        cursor.execute("ALTER TABLE tickets ADD COLUMN category_confidence REAL")
    if "priority_confidence" not in existing_cols:
        cursor.execute("ALTER TABLE tickets ADD COLUMN priority_confidence REAL")
    if "assigned_agent_id" not in existing_cols:
        cursor.execute(
            "ALTER TABLE tickets ADD COLUMN assigned_agent_id INTEGER "
            "REFERENCES agents(id) ON DELETE SET NULL"
        )
    cursor.executemany(
        "INSERT OR IGNORE INTO agents (name, department, is_active) VALUES (?, ?, 1)",
        DEMO_AGENTS,
    )
    conn.commit()

    # Pre-populate database with dataset sample if database is newly created
    cursor.execute("SELECT COUNT(*) FROM tickets")
    count = cursor.fetchone()[0]

    if count == 0:
        dataset_csv = os.path.join(BASE_DIR, "dataset", "tickets.csv")
        if os.path.exists(dataset_csv):
            import pandas as pd
            df = pd.read_csv(dataset_csv).head(25)
            predictor = get_predictor()

            for _, row in df.iterrows():
                pred = predictor.predict(str(row['ticket_text']))
                cursor.execute("""
                    INSERT INTO tickets (ticket_id, ticket_text, category, priority, department, confidence, category_confidence, priority_confidence, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(row['ticket_id']),
                    str(row['ticket_text']),
                    pred.get('category', str(row['category'])),
                    pred.get('priority', str(row['priority'])),
                    pred.get('department', str(row['department'])),
                    pred.get('category_confidence', pred.get('confidence', 95.0)),
                    pred.get('category_confidence', 95.0),
                    pred.get('priority_confidence', 80.0),
                    'Open',
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ))
            conn.commit()

    conn.close()


# Ensure DB schema is initialized on import/startup
init_db()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/admin/tickets')
def view_all_tickets():
    conn = get_db_connection()
    try:
        tickets = conn.execute(
            "SELECT t.ticket_id, t.ticket_text, t.category, t.priority, t.department, "
            "t.created_at, COALESCE(a.name, 'Unassigned') AS assigned_agent "
            "FROM tickets AS t LEFT JOIN agents AS a ON a.id = t.assigned_agent_id "
            "ORDER BY t.created_at DESC, t.id DESC"
        ).fetchall()
    finally:
        conn.close()
    return render_template('tickets.html', tickets=tickets)


def _validated_request_ticket_text():
    data = request.get_json(silent=True)
    value = data.get('ticket_text') if isinstance(data, dict) else None
    validation = validate_ticket_text(value)
    if not validation.is_valid:
        return None, (jsonify({"status": "error", "message": validation.message}), validation.status_code)
    return validation.ticket_text, None


@app.route('/api/predict', methods=['POST'])
def api_predict():
    ticket_text, error_response = _validated_request_ticket_text()
    if error_response:
        return error_response
    pred = predict_ticket(ticket_text)
    return jsonify(pred)


@app.route('/api/tickets/submit', methods=['POST'])
def submit_ticket():
    ticket_text, error_response = _validated_request_ticket_text()
    if error_response:
        return error_response

    # Execute NLP Classification & Department Routing
    pred = predict_ticket(ticket_text)

    if pred.get('status') == 'error':
        return jsonify({"status": "error", "message": pred.get('error', 'Classification failed.')}), 500

    # Generate Ticket ID
    ticket_id = f"TICK-{uuid.uuid4().hex[:6].upper()}"
    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cat_conf = pred.get('category_confidence', pred.get('confidence', 0.0))
    pri_conf = pred.get('priority_confidence', 0.0)

    # Select and save the assignment with the ticket in one transaction.
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        agent = cursor.execute(
            "SELECT id, name FROM agents WHERE department = ? AND is_active = 1 "
            "ORDER BY RANDOM() LIMIT 1",
            (pred['department'],),
        ).fetchone()
        cursor.execute("""
            INSERT INTO tickets (ticket_id, ticket_text, category, priority, department, confidence, category_confidence, priority_confidence, status, created_at, assigned_agent_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticket_id,
            ticket_text,
            pred['category'],
            pred['priority'],
            pred['department'],
            cat_conf,
            cat_conf,
            pri_conf,
            'Open',
            created_at,
            agent['id'] if agent else None,
        ))
        conn.commit()
    except sqlite3.Error:
        conn.rollback()
        app.logger.exception("Database error while saving ticket %s", ticket_id)
        return jsonify({"status": "error", "message": "The ticket could not be saved. Please try again."}), 500
    finally:
        conn.close()

    return jsonify({
        "status": "success",
        "ticket_id": ticket_id,
        "ticket_text": ticket_text,
        "category": pred['category'],
        "category_confidence": cat_conf,
        "priority": pred['priority'],
        "priority_confidence": pri_conf,
        "department": pred['department'],
        "assigned_agent": agent['name'] if agent else "Unassigned",
        "confidence": cat_conf,
        "created_at": created_at,
        "ticket_status": "Open"
    })


@app.route('/api/admin/stats', methods=['GET'])
def admin_stats():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Total Tickets
    cursor.execute("SELECT COUNT(*) FROM tickets")
    total_tickets = cursor.fetchone()[0]

    # Category breakdown
    cursor.execute("SELECT category, COUNT(*) as count FROM tickets GROUP BY category")
    category_counts = {row['category']: row['count'] for row in cursor.fetchall()}

    # Priority breakdown
    cursor.execute("SELECT priority, COUNT(*) as count FROM tickets GROUP BY priority")
    priority_counts = {row['priority']: row['count'] for row in cursor.fetchall()}

    # Department breakdown
    cursor.execute("SELECT department, COUNT(*) as count FROM tickets GROUP BY department")
    department_counts = {row['department']: row['count'] for row in cursor.fetchall()}

    # Recent tickets
    cursor.execute("""
        SELECT t.*, COALESCE(a.name, 'Unassigned') AS assigned_agent
        FROM tickets AS t
        LEFT JOIN agents AS a ON a.id = t.assigned_agent_id
        ORDER BY t.id DESC LIMIT 10
    """)
    recent_rows = cursor.fetchall()
    recent_tickets = [dict(r) for r in recent_rows]

    # Feedback statistics
    cursor.execute("SELECT COUNT(*) FROM priority_feedback")
    total_feedback = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM priority_feedback WHERE validated = 1")
    validated_feedback_count = cursor.fetchone()[0]

    cursor.execute("SELECT * FROM priority_feedback ORDER BY id DESC LIMIT 10")
    recent_feedback = [dict(r) for r in cursor.fetchall()]

    conn.close()

    # Load Model Metrics
    metrics_path = os.path.join(BASE_DIR, "models", "model_metrics.json")
    model_metrics = {}
    try:
        import json
        with open(metrics_path, 'r', encoding='utf-8') as f:
            loaded_metrics = json.load(f)
        if isinstance(loaded_metrics, dict):
            model_metrics = loaded_metrics
    except (OSError, json.JSONDecodeError):
        app.logger.warning("Saved model metrics are unavailable or malformed: %s", metrics_path)

    return jsonify({
        "total_tickets": total_tickets,
        "category_counts": category_counts,
        "priority_counts": priority_counts,
        "department_counts": department_counts,
        "recent_tickets": recent_tickets,
        "feedback_stats": {
            "total_feedback": total_feedback,
            "validated_feedback_count": validated_feedback_count,
            "pending_validation_count": total_feedback - validated_feedback_count,
            "recent_feedback": recent_feedback
        },
        "model_metrics": model_metrics
    })


@app.route('/api/admin/tickets/<ticket_id>', methods=['DELETE'])
def delete_ticket(ticket_id):
    if not ticket_id.strip():
        return jsonify({"status": "error", "message": "Ticket ID is required."}), 400

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM tickets WHERE ticket_id = ?", (ticket_id,))
        if cursor.rowcount == 0:
            conn.rollback()
            return jsonify({"status": "error", "message": f"Ticket '{ticket_id}' was not found."}), 404

        conn.commit()
        return jsonify({
            "status": "success",
            "message": f"Ticket '{ticket_id}' deleted successfully.",
            "ticket_id": ticket_id,
        })
    except sqlite3.Error:
        if conn:
            conn.rollback()
        app.logger.exception("Database error while deleting ticket %s", ticket_id)
        return jsonify({"status": "error", "message": "The ticket could not be deleted. Please try again."}), 500
    finally:
        if conn:
            conn.close()


@app.route('/api/admin/tickets/<ticket_id>/assign', methods=['POST'])
def assign_ticket_agent(ticket_id):
    conn = None
    try:
        conn = get_db_connection()
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.cursor()
        ticket = cursor.execute(
            "SELECT id, department, assigned_agent_id FROM tickets WHERE ticket_id = ?",
            (ticket_id,),
        ).fetchone()
        if not ticket:
            conn.rollback()
            return jsonify({"status": "error", "message": f"Ticket '{ticket_id}' was not found."}), 404

        agents = cursor.execute(
            "SELECT id, name FROM agents WHERE department = ? AND is_active = 1",
            (ticket['department'],),
        ).fetchall()
        if not agents:
            conn.rollback()
            return jsonify({
                "status": "error",
                "message": f"No active agents are available for department '{ticket['department']}'. The ticket was not changed.",
            }), 409

        other_agents = [agent for agent in agents if agent['id'] != ticket['assigned_agent_id']]
        if ticket['assigned_agent_id'] is not None and not other_agents:
            conn.rollback()
            return jsonify({
                "status": "error",
                "message": f"Ticket '{ticket_id}' is already assigned to the only active agent in '{ticket['department']}' and cannot be reassigned.",
            }), 409

        eligible_agents = other_agents or agents
        agent = cursor.execute(
            "SELECT id, name FROM agents WHERE id IN ("
            + ",".join("?" for _ in eligible_agents)
            + ") ORDER BY RANDOM() LIMIT 1",
            tuple(item['id'] for item in eligible_agents),
        ).fetchone()
        if not agent:
            conn.rollback()
            return jsonify({"status": "error", "message": "No eligible active agent could be selected. The ticket was not changed."}), 409

        cursor.execute(
            "UPDATE tickets SET assigned_agent_id = ? WHERE id = ?",
            (agent['id'], ticket['id']),
        )
        conn.commit()
        return jsonify({
            "status": "success",
            "ticket_id": ticket_id,
            "department": ticket['department'],
            "assigned_agent_id": agent['id'],
            "assigned_agent": agent['name'],
            "message": f"Ticket '{ticket_id}' assigned to {agent['name']}.",
        })
    except sqlite3.Error:
        if conn:
            conn.rollback()
        app.logger.exception("Database error while assigning ticket %s", ticket_id)
        return jsonify({"status": "error", "message": "The ticket could not be assigned. Please try again."}), 500
    finally:
        if conn:
            conn.close()


@app.route('/api/feedback/priority', methods=['POST'])
def submit_priority_feedback():
    data = request.get_json() or {}
    ticket_id = data.get('ticket_id', '').strip()
    corrected_priority = data.get('corrected_priority', '').strip()
    reason = data.get('reason', '').strip()
    reviewer = data.get('reviewer', 'agent').strip() or 'agent'
    validated = 1 if data.get('validated') in [1, True, '1', 'true'] else 0

    if not ticket_id:
        return jsonify({"status": "error", "message": "ticket_id is required."}), 400

    valid_priorities = ['Low', 'Medium', 'High', 'Critical']
    if corrected_priority not in valid_priorities:
        return jsonify({"status": "error", "message": f"corrected_priority must be one of {valid_priorities}."}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,))
    ticket = cursor.fetchone()

    if not ticket:
        conn.close()
        return jsonify({"status": "error", "message": f"Ticket '{ticket_id}' not found."}), 404

    feedback_id = f"FB-{uuid.uuid4().hex[:6].upper()}"
    orig_priority = ticket['priority']
    orig_conf = ticket['priority_confidence'] if 'priority_confidence' in ticket.keys() else ticket['confidence']
    ticket_text = ticket['ticket_text']

    cursor.execute("""
        INSERT INTO priority_feedback (feedback_id, ticket_id, ticket_text, original_priority, corrected_priority, original_priority_confidence, reason, reviewer, validated)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        feedback_id,
        ticket_id,
        ticket_text,
        orig_priority,
        corrected_priority,
        orig_conf,
        reason,
        reviewer,
        validated
    ))
    conn.commit()
    conn.close()

    return jsonify({
        "status": "success",
        "message": "Priority feedback submitted successfully.",
        "feedback_id": feedback_id,
        "ticket_id": ticket_id,
        "original_priority": orig_priority,
        "corrected_priority": corrected_priority,
        "validated": validated
    }), 201


@app.route('/api/feedback/priority', methods=['GET'])
def get_priority_feedback():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM priority_feedback ORDER BY id DESC")
    rows = cursor.fetchall()
    feedback_list = [dict(r) for r in rows]
    conn.close()
    return jsonify({
        "status": "success",
        "count": len(feedback_list),
        "feedback": feedback_list
    })


@app.route('/api/feedback/priority/validate', methods=['POST'])
def validate_priority_feedback():
    data = request.get_json() or {}
    feedback_id = data.get('feedback_id', '').strip()
    validated = 1 if data.get('validated', True) in [1, True, '1', 'true'] else 0

    if not feedback_id:
        return jsonify({"status": "error", "message": "feedback_id is required."}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM priority_feedback WHERE feedback_id = ?", (feedback_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return jsonify({"status": "error", "message": f"Feedback '{feedback_id}' not found."}), 404

    cursor.execute("UPDATE priority_feedback SET validated = ? WHERE feedback_id = ?", (validated, feedback_id))
    conn.commit()
    conn.close()

    return jsonify({
        "status": "success",
        "message": f"Feedback '{feedback_id}' validation status updated to {validated}.",
        "feedback_id": feedback_id,
        "validated": validated
    })


@app.route('/api/admin/retrain', methods=['POST'])
def retrain_models():
    try:
        metrics = train_and_evaluate_models()
        # Reload predictor instance
        predictor = get_predictor()
        predictor.load_models()

        return jsonify({
            "status": "success",
            "message": "Models successfully retrained and reloaded!",
            "metrics": metrics
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Retraining failed: {str(e)}"}), 500


if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)

