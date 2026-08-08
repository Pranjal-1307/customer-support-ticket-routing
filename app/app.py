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
from scripts.train_models import train_and_evaluate_models

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "tickets.db")


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes SQLite database tables and populates initial sample data if empty."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id TEXT UNIQUE NOT NULL,
            ticket_text TEXT NOT NULL,
            category TEXT NOT NULL,
            priority TEXT NOT NULL,
            department TEXT NOT NULL,
            confidence REAL NOT NULL,
            status TEXT DEFAULT 'Open',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
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
                    INSERT INTO tickets (ticket_id, ticket_text, category, priority, department, confidence, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(row['ticket_id']),
                    str(row['ticket_text']),
                    pred.get('category', str(row['category'])),
                    pred.get('priority', str(row['priority'])),
                    pred.get('department', str(row['department'])),
                    pred.get('confidence', 95.0),
                    'Open',
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ))
            conn.commit()

    conn.close()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/tickets/submit', methods=['POST'])
def submit_ticket():
    data = request.get_json() or {}
    ticket_text = data.get('ticket_text', '').strip()

    if not ticket_text:
        return jsonify({"status": "error", "message": "Ticket text is required."}), 400

    # Execute NLP Classification & Department Routing
    pred = predict_ticket(ticket_text)

    if pred.get('status') == 'error':
        return jsonify({"status": "error", "message": pred.get('error', 'Classification failed.')}), 500

    # Generate Ticket ID
    ticket_id = f"TICK-{uuid.uuid4().hex[:6].upper()}"
    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save to SQLite Database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO tickets (ticket_id, ticket_text, category, priority, department, confidence, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ticket_id,
        ticket_text,
        pred['category'],
        pred['priority'],
        pred['department'],
        pred['confidence'],
        'Open',
        created_at
    ))
    conn.commit()
    conn.close()

    return jsonify({
        "status": "success",
        "ticket_id": ticket_id,
        "ticket_text": ticket_text,
        "category": pred['category'],
        "priority": pred['priority'],
        "department": pred['department'],
        "confidence": pred['confidence'],
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
    cursor.execute("SELECT * FROM tickets ORDER BY id DESC LIMIT 10")
    recent_rows = cursor.fetchall()
    recent_tickets = [dict(r) for r in recent_rows]

    conn.close()

    # Load Model Metrics
    metrics_path = os.path.join(BASE_DIR, "models", "model_metrics.json")
    model_metrics = {}
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, 'r') as f:
            model_metrics = json.load(f)

    return jsonify({
        "total_tickets": total_tickets,
        "category_counts": category_counts,
        "priority_counts": priority_counts,
        "department_counts": department_counts,
        "recent_tickets": recent_tickets,
        "model_metrics": model_metrics
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
