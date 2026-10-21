"""Customer ticket input validation and isolated submission-route tests."""

import importlib
import os
import re
import sqlite3

import pytest

from utils.ticket_validation import MAX_TICKET_LENGTH, validate_ticket_text


INVALID_TEXTS = [
    "", "   ", "\n\t", "!!!!!!", "???????", ".....", "@#$%^&*", "😀😀😀😀",
    "asdfghjkl", "qwertyuiop", "zxcvbnm", "aaaaaaaaaa", "1234567890", "asdf 1234 !!!",
    "asdf asdf asdf",
]

VALID_TEXTS = [
    "help", "please help", "Need refund", "App login fails", "not working",
    "error 404", "OTP not received", "PIN reset failed", "I cannot log into my account",
    "Payment failed", "My order has not arrived", "Order 12345 has not arrived",
    "I was charged ₹500 twice", "I paid 100 but order is missing",
    "Login fails after update v2.1", "I cannot log in!!!", "My card was charged twice :( ",
    "URGENT: account locked", "My product arrived damaged :( ",
    "A typoed, informal message still works fine.",
]


@pytest.mark.parametrize("text", INVALID_TEXTS)
def test_validator_rejects_empty_symbol_only_and_obvious_gibberish(text):
    result = validate_ticket_text(text)
    assert not result.is_valid
    assert result.message


@pytest.mark.parametrize("text", VALID_TEXTS)
def test_validator_accepts_legitimate_ticket_messages(text):
    result = validate_ticket_text(text)
    assert result.is_valid, result.message
    assert result.ticket_text == text.strip()


def test_validator_handles_type_and_length_boundaries():
    assert not validate_ticket_text(None).is_valid
    assert not validate_ticket_text(12345).is_valid
    exact_limit = ("help with order status " * (MAX_TICKET_LENGTH // 23 + 1))[:MAX_TICKET_LENGTH]
    assert validate_ticket_text(exact_limit).is_valid
    assert validate_ticket_text(exact_limit + "x").status_code == 413
    assert validate_ticket_text(" " * (MAX_TICKET_LENGTH + 1)).status_code == 413


@pytest.fixture(scope="module")
def isolated_app(tmp_path_factory):
    """Import the Flask module with its import-time DB initialization redirected to a temp DB."""
    temp_dir = tmp_path_factory.mktemp("ticket-validation-db")
    test_db = temp_dir / "tickets.sqlite"
    real_db = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tickets.db"))

    # A baseline row prevents app import's existing empty-database sample seeding.
    with sqlite3.connect(test_db) as conn:
        conn.execute("""
            CREATE TABLE tickets (
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute(
            "INSERT INTO tickets (ticket_id,ticket_text,category,priority,department,confidence) "
            "VALUES ('TEST-BASELINE','existing record','Billing','Low','Billing Team',90)"
        )

    real_connect = sqlite3.connect

    def isolated_connect(database, *args, **kwargs):
        if os.path.normcase(os.path.abspath(os.fspath(database))) == os.path.normcase(real_db):
            database = str(test_db)
        return real_connect(database, *args, **kwargs)

    sqlite3.connect = isolated_connect
    try:
        app_module = importlib.import_module("app.app")
    finally:
        sqlite3.connect = real_connect

    app_module.DB_PATH = str(test_db)
    app_module.app.config.update(TESTING=True)
    return app_module, test_db


def _ticket_count(db_path):
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]


def _prediction(text):
    return {
        "status": "success", "ticket_text": text, "category": "Billing",
        "category_confidence": 91.0, "priority": "Medium",
        "priority_confidence": 82.0, "priority_signals": {},
        "priority_explanation": "Medium priority: Standard inquiry.",
        "department": "Billing Team", "confidence": 91.0, "error": None,
    }


@pytest.mark.parametrize("text", INVALID_TEXTS)
def test_submit_rejects_invalid_text_without_prediction_or_insert(text, isolated_app, monkeypatch):
    app_module, db_path = isolated_app
    before = _ticket_count(db_path)
    monkeypatch.setattr(app_module, "predict_ticket", lambda _: pytest.fail("prediction ran for invalid input"))

    response = app_module.app.test_client().post(
        "/api/tickets/submit", json={"ticket_text": text}
    )

    assert response.status_code == 400
    assert response.get_json()["status"] == "error"
    assert response.get_json()["message"]
    assert _ticket_count(db_path) == before


@pytest.mark.parametrize("payload", [{}, {"ticket_text": 123}, {"ticket_text": None}])
def test_malformed_or_missing_fields_are_client_errors(payload, isolated_app, monkeypatch):
    app_module, db_path = isolated_app
    before = _ticket_count(db_path)
    monkeypatch.setattr(app_module, "predict_ticket", lambda _: pytest.fail("prediction ran for malformed input"))

    response = app_module.app.test_client().post("/api/tickets/submit", json=payload)

    assert response.status_code == 400
    assert response.get_json()["status"] == "error"
    assert _ticket_count(db_path) == before


def test_malformed_json_is_a_client_error(isolated_app, monkeypatch):
    app_module, db_path = isolated_app
    before = _ticket_count(db_path)
    monkeypatch.setattr(app_module, "predict_ticket", lambda _: pytest.fail("prediction ran for malformed JSON"))

    response = app_module.app.test_client().post(
        "/api/tickets/submit", data="{", content_type="application/json"
    )

    assert response.status_code == 400
    assert response.get_json()["status"] == "error"
    assert _ticket_count(db_path) == before


def test_overlength_route_request_is_rejected_before_prediction(isolated_app, monkeypatch):
    app_module, db_path = isolated_app
    before = _ticket_count(db_path)
    monkeypatch.setattr(app_module, "predict_ticket", lambda _: pytest.fail("prediction ran for overlength input"))

    response = app_module.app.test_client().post(
        "/api/tickets/submit", json={"ticket_text": "help " * 1100}
    )

    assert response.status_code == 413
    assert response.get_json()["status"] == "error"
    assert _ticket_count(db_path) == before


def test_valid_submit_is_trimmed_saved_once_and_following_valid_request_works(isolated_app, monkeypatch):
    app_module, db_path = isolated_app
    client = app_module.app.test_client()
    calls = []
    monkeypatch.setattr(app_module, "predict_ticket", lambda text: calls.append(text) or _prediction(text))
    before = _ticket_count(db_path)

    invalid = client.post("/api/tickets/submit", json={"ticket_text": "asdf 1234 !!!"})
    assert invalid.status_code == 400
    assert _ticket_count(db_path) == before

    response = client.post("/api/tickets/submit", json={"ticket_text": "  App login fails  "})
    data = response.get_json()
    assert response.status_code == 200
    assert data["status"] == "success"
    assert data["ticket_text"] == "App login fails"
    assert calls == ["App login fails"]
    assert _ticket_count(db_path) == before + 1
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT ticket_text FROM tickets WHERE ticket_id = ?", (data["ticket_id"],)
        ).fetchall()
    assert rows == [("App login fails",)]


def test_predict_endpoint_validates_before_calling_prediction(isolated_app, monkeypatch):
    app_module, db_path = isolated_app
    calls = []
    monkeypatch.setattr(app_module, "predict_ticket", lambda text: calls.append(text) or _prediction(text))
    client = app_module.app.test_client()
    before = _ticket_count(db_path)

    invalid = client.post("/api/predict", json={"ticket_text": "😀😀😀😀"})
    assert invalid.status_code == 400
    assert calls == []
    assert _ticket_count(db_path) == before

    valid = client.post("/api/predict", json={"ticket_text": "\n Payment failed  "})
    assert valid.status_code == 200
    assert valid.get_json()["status"] == "success"
    assert calls == ["Payment failed"]
    assert _ticket_count(db_path) == before


def test_delete_route_removes_only_selected_ticket_and_keeps_feedback(isolated_app):
    app_module, db_path = isolated_app
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO tickets (ticket_id,ticket_text,category,priority,department,confidence) "
            "VALUES ('TICK-DELETE','target ticket','Billing','Medium','Billing Team',80)"
        )
        conn.execute(
            "INSERT INTO tickets (ticket_id,ticket_text,category,priority,department,confidence) "
            "VALUES ('TICK-KEEP','unrelated ticket','Shipping','Low','Logistics Team',75)"
        )
        conn.execute(
            "INSERT INTO priority_feedback (feedback_id,ticket_id,ticket_text,original_priority,corrected_priority) "
            "VALUES ('FB-KEEP','TICK-DELETE','target ticket','Medium','High')"
        )

    client = app_module.app.test_client()
    tickets_page = client.get('/admin/tickets')
    assert tickets_page.status_code == 200
    page_html = tickets_page.get_data(as_text=True)
    with sqlite3.connect(db_path) as conn:
        ticket_count = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
    assert page_html.count('class="btn btn-sm btn-outline-danger rounded-pill delete-ticket-btn"') == ticket_count
    assert 'data-ticket-id="TICK-DELETE"' in page_html
    assert 'data-ticket-id="TICK-KEEP"' in page_html

    response = client.delete('/api/admin/tickets/TICK-DELETE')

    assert response.status_code == 200
    assert response.get_json()["status"] == "success"
    with sqlite3.connect(db_path) as conn:
        tickets = {row[0] for row in conn.execute("SELECT ticket_id FROM tickets")}
        feedback = conn.execute(
            "SELECT ticket_id FROM priority_feedback WHERE feedback_id = 'FB-KEEP'"
        ).fetchone()
    assert "TICK-DELETE" not in tickets
    assert "TICK-KEEP" in tickets
    assert "TEST-BASELINE" in tickets
    assert feedback == ("TICK-DELETE",)

    correction = app_module.app.test_client().post('/api/feedback/priority', json={
        "ticket_id": "TICK-KEEP",
        "corrected_priority": "High",
        "reason": "Regression check for the existing Correct action.",
    })
    assert correction.status_code == 201
    assert correction.get_json()["status"] == "success"

    stats = app_module.app.test_client().get('/api/admin/stats').get_json()
    recent_ids = {ticket["ticket_id"] for ticket in stats["recent_tickets"]}
    assert "TICK-DELETE" not in recent_ids
    assert "TICK-KEEP" in recent_ids
    refreshed_page = client.get('/admin/tickets').get_data(as_text=True)
    assert 'TICK-DELETE' not in refreshed_page
    assert 'TICK-KEEP' in refreshed_page


def test_delete_route_handles_missing_ticket_and_database_error(isolated_app, monkeypatch):
    app_module, db_path = isolated_app
    before = _ticket_count(db_path)
    client = app_module.app.test_client()

    missing = client.delete('/api/admin/tickets/TICK-MISSING')
    assert missing.status_code == 404
    assert missing.get_json()["status"] == "error"
    assert _ticket_count(db_path) == before

    monkeypatch.setattr(app_module, "get_db_connection", lambda: (_ for _ in ()).throw(sqlite3.OperationalError("test failure")))
    failed = client.delete('/api/admin/tickets/TICK-ANY')
    assert failed.status_code == 500
    assert failed.get_json()["status"] == "error"
    assert "test failure" not in failed.get_json()["message"]


def test_all_tickets_delete_action_uses_existing_confirmed_delete_flow():
    root = os.path.dirname(os.path.dirname(__file__))
    with open(os.path.join(root, "app", "templates", "tickets.html"), encoding="utf-8") as file:
        template = file.read()
    with open(os.path.join(root, "app", "static", "script.js"), encoding="utf-8") as file:
        script = file.read()

    assert 'class="btn btn-sm btn-outline-danger rounded-pill delete-ticket-btn"' in template
    assert "event.target.closest('.delete-ticket-btn')" in script
    assert "deleteTicket(deleteButton.dataset.ticketId, deleteButton)" in script
    delete_start = script.index("async function deleteTicket(")
    delete_flow = script[delete_start:]
    assert "window.confirm(`Delete ticket ${ticketId}?" in delete_flow
    assert "fetch(`/api/admin/tickets/${encodeURIComponent(ticketId)}`" in delete_flow
    assert delete_flow.index("if (!response.ok || data.status !== 'success')") < delete_flow.index("row.remove()")


def test_admin_can_assign_and_reassign_without_changing_ticket_fields(isolated_app):
    app_module, db_path = isolated_app
    ticket_id = "TICK-ASSIGN-REPEAT"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO tickets (ticket_id,ticket_text,category,priority,department,confidence,created_at) "
            "VALUES (?, 'billing issue', 'Billing', 'High', 'Billing Team', 88, '2026-10-21 10:00:00')",
            (ticket_id,),
        )
        original = conn.execute(
            "SELECT ticket_text,category,priority,department,confidence,created_at FROM tickets WHERE ticket_id = ?",
            (ticket_id,),
        ).fetchone()

    client = app_module.app.test_client()
    assigned = client.post(f"/api/admin/tickets/{ticket_id}/assign")
    assert assigned.status_code == 200
    first_agent = assigned.get_json()["assigned_agent"]
    assert first_agent.startswith("Demo Agent A")

    with sqlite3.connect(db_path) as conn:
        first = conn.execute(
            "SELECT t.ticket_text,t.category,t.priority,t.department,t.confidence,t.created_at,t.assigned_agent_id,a.name,a.department,a.is_active "
            "FROM tickets t LEFT JOIN agents a ON a.id=t.assigned_agent_id WHERE t.ticket_id = ?",
            (ticket_id,),
        ).fetchone()
    assert first[:6] == original
    assert first[7] == first_agent and first[8] == "Billing Team" and first[9] == 1

    reassigned = client.post(f"/api/admin/tickets/{ticket_id}/assign")
    assert reassigned.status_code == 200
    second_agent = reassigned.get_json()["assigned_agent"]
    assert second_agent != first_agent
    with sqlite3.connect(db_path) as conn:
        second = conn.execute(
            "SELECT ticket_text,category,priority,department,confidence,created_at FROM tickets WHERE ticket_id = ?",
            (ticket_id,),
        ).fetchone()
    assert second == original

    recent = client.get("/api/admin/stats").get_json()["recent_tickets"]
    assert next(row for row in recent if row["ticket_id"] == ticket_id)["assigned_agent"] == second_agent
    all_tickets = client.get("/admin/tickets").get_data(as_text=True)
    assert second_agent in all_tickets
    assert "Reassign Agent" in all_tickets
    assert "assign-agent-btn" in all_tickets


def test_admin_assignment_handles_one_agent_no_agents_and_missing_ticket(isolated_app):
    app_module, db_path = isolated_app
    ticket_id = "TICK-ASSIGN-LIMIT"
    with sqlite3.connect(db_path) as conn:
        eligible = conn.execute(
            "SELECT id FROM agents WHERE department='Billing Team' ORDER BY id LIMIT 2"
        ).fetchall()
        assert len(eligible) == 2
        conn.execute(
            "INSERT INTO tickets (ticket_id,ticket_text,category,priority,department,confidence,assigned_agent_id) "
            "VALUES (?, 'billing issue', 'Billing', 'Medium', 'Billing Team', 80, ?)",
            (ticket_id, eligible[0][0]),
        )
        conn.execute("UPDATE agents SET is_active=0 WHERE id=?", (eligible[1][0],))

    client = app_module.app.test_client()
    one_agent = client.post(f"/api/admin/tickets/{ticket_id}/assign")
    assert one_agent.status_code == 409
    assert "only active agent" in one_agent.get_json()["message"]
    with sqlite3.connect(db_path) as conn:
        assignment = conn.execute(
            "SELECT assigned_agent_id FROM tickets WHERE ticket_id=?", (ticket_id,)
        ).fetchone()[0]
        assert assignment == eligible[0][0]
        conn.execute("UPDATE agents SET is_active=0 WHERE id=?", (eligible[0][0],))

    no_agents = client.post(f"/api/admin/tickets/{ticket_id}/assign")
    assert no_agents.status_code == 409
    assert "No active agents" in no_agents.get_json()["message"]
    missing = client.post("/api/admin/tickets/TICK-DOES-NOT-EXIST/assign")
    assert missing.status_code == 404
    with sqlite3.connect(db_path) as conn:
        assert conn.execute(
            "SELECT assigned_agent_id FROM tickets WHERE ticket_id=?", (ticket_id,)
        ).fetchone()[0] == eligible[0][0]


def test_quick_samples_and_existing_error_display_remain_present():
    root = os.path.dirname(os.path.dirname(__file__))
    with open(os.path.join(root, "app", "templates", "index.html"), encoding="utf-8") as file:
        html = file.read()
    section = html.split("<!-- Quick Sample Buttons -->", 1)[1].split("<!-- Ticket Form -->", 1)[0]
    buttons = re.findall(r'<button\b[^>]*class="[^"]*sample-btn[^"]*"[^>]*data-sample="[^"]+"[^>]*>.*?</button>', section, re.S)
    assert len(buttons) == 8
    assert all('type="button"' in button for button in buttons)
    with open(os.path.join(root, "app", "static", "script.js"), encoding="utf-8") as file:
        script = file.read()
    assert "e.currentTarget.getAttribute('data-sample')" in script
    assert "alert(`Error: ${data.message" in script
    assert "Please enter a ticket description." in script
    assert "window.confirm(`Delete ticket ${ticketId}?" in script
    assert "fetch(`/api/admin/tickets/${encodeURIComponent(ticketId)}`" in script
    assert "await loadAdminStats()" in script
    assert "openFeedbackModal('${t.ticket_id}'" in script
    assert ">Correct" in script
    assert "btn-outline-danger" in script
