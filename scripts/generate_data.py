"""
Synthetic Dataset Generator for Customer Support Ticket Routing System.
Generates 5000+ realistic customer support tickets with categories, priorities, and assigned departments.
Saves tickets.csv, train.csv, and test.csv using stratified splitting.
"""

import os
import random
import uuid
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.route import get_department_for_category

# Base Seed for Reproducibility
random.seed(42)

CATEGORIES = [
    "Billing",
    "Technical Support",
    "Refund",
    "Shipping",
    "Account",
    "Complaint",
    "Product Inquiry",
    "Cancellation"
]

PRIORITIES = ["Low", "Medium", "High", "Critical"]

# Sentence components for realistic synthetic tickets
TICKET_PATTERNS = {
    "Billing": {
        "phrases": [
            "My credit card was charged twice for transaction {id}.",
            "I noticed an unauthorized charge of ${amount} on my monthly invoice.",
            "Why was I billed ${amount} when my plan is ${plan_amount}?",
            "I need an official tax invoice and payment receipt for order {id}.",
            "My payment failed yesterday, but money was deducted from my bank account.",
            "There is a billing discrepancy on my recent statement for billing period {period}.",
            "I received a late payment penalty charge even though I paid on time.",
            "Can you update my credit card billing address and tax ID for billing?",
            "I am being billed for a service that was supposed to be free during trial.",
            "Double charge observed on my bank statement for invoice {id}."
        ],
        "priority_weights": [0.1, 0.4, 0.4, 0.1]  # Low, Med, High, Critical
    },
    "Technical Support": {
        "phrases": [
            "My internet is not working and the router LED is blinking red.",
            "The web application keeps throwing 500 internal server errors when submitting forms.",
            "System crashed completely during export task showing error code {err_code}.",
            "Unable to connect to remote database server from IP address {ip}.",
            "The desktop application freezes every time I click the save button.",
            "Mobile app keeps crashing on launch after updating to latest iOS/Android version.",
            "Page load times are extremely slow and API calls are timing out.",
            "VPN connection drops every 5 minutes unexpectedly.",
            "Software installation fails due to missing DLL files on Windows 11.",
            "Bluetooth sync is failing between the hardware device and dashboard."
        ],
        "priority_weights": [0.1, 0.3, 0.4, 0.2]
    },
    "Refund": {
        "phrases": [
            "I requested a refund for order {id} two weeks ago but haven't received it.",
            "Item arrived damaged in transit. I want an immediate full refund.",
            "The product quality is sub-par and does not match description. Please refund.",
            "I accidentally purchased the wrong subscription tier and need money back.",
            "Service was down during our event. Requesting credit refund to account.",
            "Returned package was delivered to warehouse. When will refund be processed?",
            "Order was canceled by system but refund has not posted to my account.",
            "I demand a prompt refund for defective goods received yesterday.",
            "Partial refund missing from my previous dispute case {id}.",
            "Charged after cancellation request, need immediate reimbursement."
        ],
        "priority_weights": [0.2, 0.4, 0.3, 0.1]
    },
    "Shipping": {
        "phrases": [
            "Where is my package? Tracking number {id} shows no movement for 5 days.",
            "Estimated delivery date has passed but package has not arrived.",
            "Tracking indicates delivered, but package is missing from my doorstep.",
            "My order {id} was shipped to wrong delivery address.",
            "Customs clearance is delaying package shipment at border transit.",
            "I need to change shipping address for order {id} before dispatch.",
            "Package arrived with crushed outer box and broken contents.",
            "Express shipping paid, but package sent via standard economy post.",
            "Delivery courier left parcel in rain without notification.",
            "International shipment stuck at distribution center for over a week."
        ],
        "priority_weights": [0.2, 0.5, 0.2, 0.1]
    },
    "Account": {
        "phrases": [
            "I cannot log into my account despite entering correct password.",
            "Two-factor authentication code is not being received on my phone.",
            "My account has been locked due to multiple failed login attempts.",
            "I need to update my registered email address and phone number.",
            "Received suspicious password reset email without requesting it.",
            "Unable to reset password via email link; link expired immediately.",
            "How do I transfer account ownership to another administrator?",
            "My profile settings fail to save changes.",
            "Single Sign-On (SSO) integration is failing for company domain.",
            "Account permissions are misconfigured and user cannot access dashboard."
        ],
        "priority_weights": [0.1, 0.3, 0.5, 0.1]
    },
    "Complaint": {
        "phrases": [
            "Very dissatisfied with rude customer service representative on call.",
            "Agent closed my support ticket without resolving the actual issue.",
            "Escalating issue: product failed during critical presentation.",
            "Worst service experience ever! Unprofessional support team.",
            "No response to urgent tickets for 72 hours despite premium support SLA.",
            "System downtime caused business revenue loss for our team.",
            "Misleading advertising regarding feature capabilities on sales page.",
            "Repeatedly transferred between agents without any resolution.",
            "Formal complaint regarding poor product build quality and support delay.",
            "Appalled by lack of response to critical security vulnerability report."
        ],
        "priority_weights": [0.1, 0.3, 0.4, 0.2]
    },
    "Product Inquiry": {
        "phrases": [
            "Does your enterprise plan support custom API integrations?",
            "What are the hardware specifications required to run this software?",
            "Can you provide information on bulk licensing discounts for 500+ users?",
            "Is this product compatible with macOS Sequoia and Apple Silicon?",
            "Where can I find user manuals and documentation for model {model}?",
            "Are there any upcoming features planned for mobile tablet views?",
            "Can I try out premium features during a free trial demo?",
            "What is the warranty period and coverage for international orders?",
            "Does the cloud platform comply with SOC2 and HIPAA standards?",
            "Looking for comparison between standard and professional tiers."
        ],
        "priority_weights": [0.6, 0.3, 0.1, 0.0]
    },
    "Cancellation": {
        "phrases": [
            "I want to cancel my subscription effective immediately.",
            "Please turn off auto-renewal for my annual membership plan.",
            "Need assistance closing and permanently deleting my account.",
            "Canceling contract due to budget constraints in our organization.",
            "We are switching to an alternative solution and wish to terminate service.",
            "How do I cancel my order before it enters fulfillment phase?",
            "Subscription cancellation button is missing from account settings page.",
            "Please confirm cancellation of recurring billing for account {id}.",
            "Requesting account closure and deletion of stored customer data.",
            "Closing subscription due to missing essential features."
        ],
        "priority_weights": [0.2, 0.5, 0.2, 0.1]
    }
}

ADDITIONAL_NOISE = [
    " Please advise as soon as possible.",
    " Thank you for your assistance.",
    " Urgent attention required.",
    " Contact me via email or phone.",
    " Appreciate a quick update on this.",
    " Kindly look into this matter promptly.",
    " Looking forward to your prompt response.",
    ""
]


def generate_tickets(target_count=5200):
    tickets = []
    per_category = target_count // len(CATEGORIES)

    ticket_counter = 1001

    for category in CATEGORIES:
        info = TICKET_PATTERNS[category]
        phrases = info["phrases"]
        priority_weights = info["priority_weights"]

        for _ in range(per_category):
            ticket_id = f"TICK-{ticket_counter:06d}"
            ticket_counter += 1

            # Select phrase and fill parameters
            phrase_tmpl = random.choice(phrases)
            text = phrase_tmpl.format(
                id=random.randint(10000, 99999),
                amount=random.choice([19.99, 49.00, 99.99, 249.50, 499.00]),
                plan_amount=random.choice([9.99, 29.99, 79.99]),
                period=random.choice(["Jan 2026", "Feb 2026", "Q1 2026"]),
                err_code=random.choice(["ERR_502", "NULL_POINTER_EXC", "SOCKET_TIMEOUT", "AUTH_DENIED"]),
                ip=f"192.168.{random.randint(1,254)}.{random.randint(1,254)}",
                model=f"PRO-{random.randint(100,999)}"
            )

            # Add minor noise variation to ensure uniqueness
            noise = random.choice(ADDITIONAL_NOISE)
            full_text = text + noise

            # Select priority based on weighted distribution
            priority = random.choices(PRIORITIES, weights=priority_weights, k=1)[0]
            department = get_department_for_category(category)

            tickets.append({
                "ticket_id": ticket_id,
                "ticket_text": full_text,
                "category": category,
                "priority": priority,
                "department": department,
                "source": "synthetic"
            })

    # Shuffle
    random.shuffle(tickets)
    df = pd.DataFrame(tickets)
    return df


def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")
    raw_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    print("Generating synthetic dataset (5200 samples)...")
    df = generate_tickets(5200)

    # Save to raw directory and legacy dataset directory
    raw_synthetic_path = os.path.join(raw_dir, "synthetic_tickets.csv")
    df.to_csv(raw_synthetic_path, index=False)
    print(f"Saved raw synthetic dataset to {raw_synthetic_path} (Shape: {df.shape})")

    tickets_path = os.path.join(dataset_dir, "tickets.csv")
    df.to_csv(tickets_path, index=False)
    print(f"Saved dataset copy to {tickets_path} (Shape: {df.shape})")

    # Stratified Train/Test split (80% train, 20% test)
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['category']
    )

    train_path = os.path.join(dataset_dir, "train.csv")
    test_path = os.path.join(dataset_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved train dataset to {train_path} (Shape: {train_df.shape})")
    print(f"Saved test dataset to {test_path} (Shape: {test_df.shape})")


if __name__ == "__main__":
    main()
