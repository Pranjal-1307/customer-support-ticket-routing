"""
Main Entry Point for Customer Support Ticket Routing System using NLP.
Executes data setup, model training check, database initialization, and starts Flask web application.
"""

import os
import sys

# Add project root to Python search path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from scripts.generate_data import main as generate_dataset
from scripts.train_models import train_and_evaluate_models
from app.app import app, init_db


def main():
    print("=" * 70)
    print(" CUSTOMER SUPPORT TICKET ROUTING SYSTEM USING NLP")
    print("=" * 70)

    # 1. Ensure dataset and splits exist
    tickets_path = os.path.join(PROJECT_ROOT, "dataset", "tickets.csv")
    train_path = os.path.join(PROJECT_ROOT, "dataset", "train.csv")
    test_path = os.path.join(PROJECT_ROOT, "dataset", "test.csv")
    if not os.path.exists(tickets_path) or not os.path.exists(train_path) or not os.path.exists(test_path):
        print("\n[INFO] Dataset or splits not found. Generating and preparing synthetic dataset...")
        generate_dataset()
    else:
        print(f"\n[OK] Dataset and splits found at {tickets_path}")

    # 2. Ensure trained models exist
    model_path = os.path.join(PROJECT_ROOT, "models", "ticket_classifier.pkl")
    if not os.path.exists(model_path):
        print("\n[INFO] Model artifacts not found. Training and evaluating ML classifiers...")
        train_and_evaluate_models()
    else:
        print(f"[OK] Trained ML models found at {model_path}")

    # 3. Initialize SQLite Database
    print("\n[INFO] Initializing SQLite database...")
    init_db()
    print("[OK] Database ready at tickets.db")

    # 4. Start Flask Server
    print("\n[INFO] Launching Web Application on http://127.0.0.1:5000 ...")
    print("Press Ctrl+C to stop server.\n")
    app.run(host='127.0.0.1', port=5000, debug=False)


if __name__ == "__main__":
    main()
