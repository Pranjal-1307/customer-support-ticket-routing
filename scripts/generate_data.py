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
            ("My credit card was charged twice for transaction {id}.", "High"),
            ("I noticed an unauthorized charge of ${amount} on my monthly invoice.", "Critical"),
            ("Why was I billed ${amount} when my plan is ${plan_amount}?", "Medium"),
            ("I need an official tax invoice and payment receipt for order {id}.", "Low"),
            ("My payment failed yesterday, but money was deducted from my bank account.", "High"),
            ("There is a billing discrepancy on my recent statement for billing period {period}.", "Medium"),
            ("I received a late payment penalty charge even though I paid on time.", "Medium"),
            ("Can you update my credit card billing address and tax ID for billing?", "Low"),
            ("I am being billed for a service that was supposed to be free during trial.", "Medium"),
            ("Double charge observed on my bank statement for invoice {id}.", "High")
        ]
    },
    "Technical Support": {
        "phrases": [
            ("My internet is not working and the router LED is blinking red.", "High"),
            ("The web application keeps throwing 500 internal server errors when submitting forms.", "High"),
            ("System crashed completely during export task showing error code {err_code}.", "Critical"),
            ("Unable to connect to remote database server from IP address {ip}.", "Critical"),
            ("The desktop application freezes every time I click the save button.", "High"),
            ("Mobile app keeps crashing on launch after updating to latest iOS/Android version.", "High"),
            ("Page load times are extremely slow and API calls are timing out.", "Medium"),
            ("VPN connection drops every 5 minutes unexpectedly.", "High"),
            ("Software installation fails due to missing DLL files on Windows 11.", "Medium"),
            ("Bluetooth sync is failing between the hardware device and dashboard.", "Medium")
        ]
    },
    "Refund": {
        "phrases": [
            ("I requested a refund for order {id} two weeks ago but haven't received it.", "Medium"),
            ("Item arrived damaged in transit. I want an immediate full refund.", "High"),
            ("The product quality is sub-par and does not match description. Please refund.", "Low"),
            ("I accidentally purchased the wrong subscription tier and need money back.", "Low"),
            ("Service was down during our event. Requesting credit refund to account.", "Medium"),
            ("Returned package was delivered to warehouse. When will refund be processed?", "Low"),
            ("Order was canceled by system but refund has not posted to my account.", "Medium"),
            ("I demand a prompt refund for defective goods received yesterday.", "High"),
            ("Partial refund missing from my previous dispute case {id}.", "Medium"),
            ("Charged after cancellation request, need immediate reimbursement.", "High")
        ]
    },
    "Shipping": {
        "phrases": [
            ("Where is my package? Tracking number {id} shows no movement for 5 days.", "Medium"),
            ("Estimated delivery date has passed but package has not arrived.", "Medium"),
            ("Tracking indicates delivered, but package is missing from my doorstep.", "High"),
            ("My order {id} was shipped to wrong delivery address.", "High"),
            ("Customs clearance is delaying package shipment at border transit.", "Low"),
            ("I need to change shipping address for order {id} before dispatch.", "High"),
            ("Package arrived with crushed outer box and broken contents.", "High"),
            ("Express shipping paid, but package sent via standard economy post.", "Medium"),
            ("Delivery courier left parcel in rain without notification.", "Medium"),
            ("International shipment stuck at distribution center for over a week.", "Medium")
        ]
    },
    "Account": {
        "phrases": [
            ("I cannot log into my account despite entering correct password.", "High"),
            ("Two-factor authentication code is not being received on my phone.", "High"),
            ("My account has been locked due to multiple failed login attempts.", "High"),
            ("I need to update my registered email address and phone number.", "Low"),
            ("Received suspicious password reset email without requesting it.", "Critical"),
            ("Unable to reset password via email link; link expired immediately.", "Medium"),
            ("How do I transfer account ownership to another administrator?", "Low"),
            ("My profile settings fail to save changes.", "Low"),
            ("Single Sign-On (SSO) integration is failing for company domain.", "High"),
            ("Account permissions are misconfigured and user cannot access dashboard.", "High")
        ]
    },
    "Complaint": {
        "phrases": [
            ("Very dissatisfied with rude customer service representative on call.", "Low"),
            ("Agent closed my support ticket without resolving the actual issue.", "Medium"),
            ("Escalating issue: product failed during critical presentation.", "High"),
            ("Worst service experience ever! Unprofessional support team.", "Low"),
            ("No response to urgent tickets for 72 hours despite premium support SLA.", "High"),
            ("System downtime caused business revenue loss for our team.", "Critical"),
            ("Misleading advertising regarding feature capabilities on sales page.", "Low"),
            ("Repeatedly transferred between agents without any resolution.", "Medium"),
            ("Formal complaint regarding poor product build quality and support delay.", "Medium"),
            ("Appalled by lack of response to critical security vulnerability report.", "Critical")
        ]
    },
    "Product Inquiry": {
        "phrases": [
            ("Does your enterprise plan support custom API integrations?", "Low"),
            ("What are the hardware specifications required to run this software?", "Low"),
            ("Can you provide information on bulk licensing discounts for 500+ users?", "Medium"),
            ("Is this product compatible with macOS Sequoia and Apple Silicon?", "Low"),
            ("Where can I find user manuals and documentation for model {model}?", "Low"),
            ("Are there any upcoming features planned for mobile tablet views?", "Low"),
            ("Can I try out premium features during a free trial demo?", "Low"),
            ("What is the warranty period and coverage for international orders?", "Low"),
            ("Does the cloud platform comply with SOC2 and HIPAA standards?", "Medium"),
            ("Looking for comparison between standard and professional tiers.", "Low")
        ]
    },
    "Cancellation": {
        "phrases": [
            ("I want to cancel my subscription effective immediately.", "High"),
            ("Please turn off auto-renewal for my annual membership plan.", "Medium"),
            ("Need assistance closing and permanently deleting my account.", "High"),
            ("Canceling contract due to budget constraints in our organization.", "Medium"),
            ("We are switching to an alternative solution and wish to terminate service.", "High"),
            ("How do I cancel my order before it enters fulfillment phase?", "Medium"),
            ("Subscription cancellation button is missing from account settings page.", "High"),
            ("Please confirm cancellation of recurring billing for account {id}.", "Medium"),
            ("Requesting account closure and deletion of stored customer data.", "High"),
            ("Closing subscription due to missing essential features.", "Medium")
        ]
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

        for _ in range(per_category):
            ticket_id = f"TICK-{ticket_counter:06d}"
            ticket_counter += 1

            # Select phrase and fill parameters
            phrase_tmpl, base_priority = random.choice(phrases)
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

            # Apply deterministic priority upgrade based on noise urgency
            if noise in [" Urgent attention required.", " Please advise as soon as possible."]:
                p_idx = PRIORITIES.index(base_priority)
                priority = PRIORITIES[min(p_idx + 1, len(PRIORITIES) - 1)]
            else:
                priority = base_priority

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

    # Run canonical preparation and split automatically to maintain consistency
    print("\n[INFO] Calling canonical dataset preparation script prepare_dataset.py...")
    from scripts.prepare_dataset import prepare_data
    prepare_data()


if __name__ == "__main__":
    main()
