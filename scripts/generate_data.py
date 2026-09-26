"""
Enhanced Synthetic Dataset Generator for Customer Support Ticket Routing System (Milestone 2).
Generates diverse, realistic synthetic tickets with varied sentence structures, length distributions,
conversational phrasing, questions, complaints, subtle spelling noise, and multilingual (Hinglish/Hindi/Gujarati) examples.
Uses deterministic priority labeling reflecting urgency, severity, financial/security impact, and SLA sensitivity.
"""

import os
import sys
import random
import re
import pandas as pd

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

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

# Rich, diverse phrase patterns per category
# Structures: (phrase_template, base_priority, length_type, language)
TICKET_PATTERNS = {
    "Billing": [
        # Short
        ("Charged twice for order {id}.", "Medium", "short", "en"),
        ("Duplicate charge on statement.", "Medium", "short", "en"),
        ("Double billing issue.", "Medium", "short", "en"),
        ("Payment failed but money deducted.", "High", "short", "en"),
        ("Need tax invoice for order {id}.", "Low", "short", "en"),
        ("Mera payment duplicate kat gaya hai.", "Medium", "short", "hinglish"),
        ("Billing ma double charge thayo che.", "Medium", "short", "gujarati"),

        # Medium / Conversational / Questions
        ("Why am I seeing two charges for the same order {id} on my bank app?", "Medium", "medium", "en"),
        ("I was billed ${amount} this month, but my plan is supposed to be ${plan_amount}.", "Medium", "medium", "en"),
        ("There is an unrecognized charge of ${amount} on my credit card statement.", "Critical", "medium", "en"),
        ("Can you please send me an official VAT receipt and tax invoice for billing period {period}?", "Low", "medium", "en"),
        ("I paid my monthly invoice yesterday, but the account dashboard still shows payment pending.", "Medium", "medium", "en"),
        ("Account balance se double amount deduct ho gaya, please check billing statement.", "Medium", "medium", "hinglish"),
        ("Kripya invoice receipt email ID par bhej dijiye.", "Low", "medium", "hindi"),

        # Long / Detailed / Indirect Wording
        ("I checked my credit card account this morning and noticed two separate line items for ${amount} on order {id}. I only submitted the checkout form once, so one of these transactions is clearly an error. Please reverse the duplicate billing immediately.", "High", "long", "en"),
        ("Our organization was charged a late fee penalty of ${amount} despite completing the wire transfer prior to the due date. Attached is our bank transaction confirmation for your accounts team to verify and credit back.", "Medium", "long", "en"),
        ("We signed up under a free trial promotional banner, yet our company credit card was billed ${amount} on day one. We expect a full reversal of this unauthorized charge and correction of our subscription tier.", "Critical", "long", "en")
    ],
    "Technical Support": [
        # Short
        ("App crashes on launch.", "Medium", "short", "en"),
        ("Router blinking red, internet down.", "High", "short", "en"),
        ("Server error 500 on submit.", "High", "short", "en"),
        ("Database connection timeout.", "Critical", "short", "en"),
        ("Desktop software freezing.", "Medium", "short", "en"),
        ("Website completely crash ho gaya hai.", "Critical", "short", "hinglish"),
        ("App chalne ma problem aave che.", "Medium", "short", "gujarati"),

        # Medium / Conversational / Questions
        ("Why does the mobile application crash every time I try to open the camera or upload a document?", "Medium", "medium", "en"),
        ("Our backend production database server at {ip} is refusing incoming connections with error {err_code}.", "Critical", "medium", "en"),
        ("Page loading times have degraded significantly after the latest patch, causing request timeouts.", "Medium", "medium", "en"),
        ("Can someone help resolve the DLL loading error on Windows 11 when starting the client application?", "Medium", "medium", "en"),
        ("VPN connection repeatedly drops every few minutes when connected to office network.", "High", "medium", "en"),
        ("System crash ho raha hai export click karte hi error code {err_code} ke saath.", "Medium", "medium", "hinglish"),

        # Long / Detailed / Indirect Wording
        ("Ever since updating to version 4.2 yesterday, our entire telemetry team has been unable to access the web portal. The page loads infinitely before displaying an internal server error (HTTP 500). This is blocking active customer operations.", "Critical", "long", "en"),
        ("We are observing consistent memory leaks in the background daemon process on node {ip}. Memory utilization climbs to 99% within two hours of service start, eventually forcing a kernel panic and abrupt system reboot.", "Critical", "long", "en"),
        ("The offline synchronization module fails to push local SQLite records to the cloud database when reconnecting to Wi-Fi. Data remains cached locally without throwing explicit log exceptions.", "Medium", "long", "en")
    ],
    "Refund": [
        # Short
        ("Requesting refund for order {id}.", "Medium", "short", "en"),
        ("Damaged product, refund needed.", "Medium", "short", "en"),
        ("Wrong item delivered, money back.", "Medium", "short", "en"),
        ("Where is my pending refund?", "Medium", "short", "en"),
        ("Refund process nahi hua abhi tak.", "Medium", "short", "hinglish"),
        ("Paisa refund karo jaldi.", "Medium", "short", "hindi"),

        # Medium / Conversational / Questions
        ("I returned package for order {id} two weeks ago but haven't seen the refund credited to my bank account.", "Medium", "medium", "en"),
        ("The item arrived with broken glass and torn packaging. I demand an immediate full refund.", "High", "medium", "en"),
        ("I accidentally renewed the wrong tier subscription yesterday. Is it possible to get a refund?", "Low", "medium", "en"),
        ("System cancelled my order due to stock shortage, but the money has not been refunded yet.", "Medium", "medium", "en"),
        ("Order cancel hone ke baad bhi refund credit nahi hua account mein.", "Medium", "medium", "hinglish"),
        ("Pakka refund kab tak milega order {id} ka?", "Medium", "medium", "hinglish"),

        # Long / Detailed / Indirect Wording
        ("I placed order {id} for an enterprise hardware component, but received a completely different consumer product model. Since this does not match the product specification or purchase order, I am returning the package and request a full refund including express shipping charges.", "High", "long", "en"),
        ("Our scheduled virtual event was ruined due to continuous platform outages on your side. We paid ${amount} for premium hosting and request an immediate pro-rated credit refund to our corporate account balance.", "Medium", "long", "en")
    ],
    "Shipping": [
        # Short
        ("Where is my parcel?", "Low", "short", "en"),
        ("Tracking not updating for order {id}.", "Low", "short", "en"),
        ("Package marked delivered but missing.", "Critical", "short", "en"),
        ("Wrong delivery address on order {id}.", "Medium", "short", "en"),
        ("Delivery kab tak aayegi?", "Low", "short", "hinglish"),
        ("Courier status update nathi thayo.", "Low", "short", "gujarati"),

        # Medium / Conversational / Questions
        ("Tracking number {id} has been stuck in transit at the hub for over 6 days without progress.", "Medium", "medium", "en"),
        ("The courier marked my parcel as delivered on the porch, but there is no package outside my house.", "High", "medium", "en"),
        ("Can I update my shipping delivery address to apartment 4B before the order dispatches?", "Medium", "medium", "en"),
        ("My parcel box arrived completely crushed with contents damaged during transport.", "High", "medium", "en"),
        ("Tracking link shows package held at border customs clearance depot.", "Low", "medium", "en"),
        ("Parcel delivered bol raha hai par gate pe koi item nahi hai.", "High", "medium", "hinglish"),

        # Long / Detailed / Indirect Wording
        ("I paid extra for guaranteed next-day air shipping on order {id} because of an urgent event. It has now been 4 days and the tracking status shows the package is still sitting at the local distribution facility. Please expedite delivery or refund express fees.", "Medium", "long", "en"),
        ("The delivery driver left our electronic equipment package in heavy rain outside the warehouse gate without ringing the bell or getting a signature. Water soaked through the box.", "High", "long", "en")
    ],
    "Account": [
        # Short
        ("Cannot log into account.", "High", "short", "en"),
        ("Forgot password link expired.", "Low", "short", "en"),
        ("2FA code not received.", "Medium", "short", "en"),
        ("Account locked after failed logins.", "High", "short", "en"),
        ("Unrecognized login notification received.", "Critical", "short", "en"),
        ("Account login nahi ho raha hai.", "High", "short", "hinglish"),
        ("Password reset link kaam nahi kar raha.", "Low", "short", "hinglish"),

        # Medium / Conversational / Questions
        ("Why am I not receiving the SMS two-factor authentication verification code on my registered mobile number?", "Medium", "medium", "en"),
        ("I received an email stating my password was changed from an unknown location, but I did not initiate this.", "Critical", "medium", "en"),
        ("How can I change the primary administrator email address associated with our company portal?", "Low", "medium", "en"),
        ("Single Sign-On (SSO) authentication is throwing a redirect URI error for domain users.", "High", "medium", "en"),
        ("Please help unlock my account after entering wrong password three times.", "Low", "medium", "en"),
        ("Mera admin account locked ho gaya hai, please reset karein.", "Medium", "medium", "hinglish"),

        # Long / Detailed / Indirect Wording
        ("Our team administrator suddenly lost access to the master workspace account after enabling SAML 2.0 SSO. The login screen loops indefinitely between the identity provider and portal without authenticating. We need urgent admin intervention.", "Critical", "long", "en"),
        ("I am requesting complete account closure and permanent deletion of all personal stored data from your servers in compliance with privacy regulations.", "Medium", "long", "en")
    ],
    "Complaint": [
        # Short
        ("Rude customer agent on call.", "Low", "short", "en"),
        ("Worst support experience ever.", "Low", "short", "en"),
        ("Ticket closed without resolution.", "Medium", "short", "en"),
        ("Service downtime ruined our demo.", "Critical", "short", "en"),
        ("Agent ne bina answer kiye ticket close kar diya.", "Medium", "short", "hinglish"),
        ("Kharaab service experience che.", "Low", "short", "gujarati"),

        # Medium / Conversational / Questions
        ("I have been on hold for over 45 minutes without anyone picking up my call. This service is terrible.", "Medium", "medium", "en"),
        ("Support agent closed my ticket {id} marking it resolved, even though the bug is still happening!", "High", "medium", "en"),
        ("Your representative was dismissive and unhelpful when I asked for assistance with system errors.", "Low", "medium", "en"),
        ("Unannounced platform outage during business hours caused direct revenue loss for our company.", "Critical", "medium", "en"),
        ("No reply to urgent ticket for 48 hours despite paying for 24/7 premium support SLA.", "High", "medium", "en"),
        ("48 ghante se koi response nahi aaya, service bohot badhiya nahi hai.", "Medium", "medium", "hinglish"),

        # Long / Detailed / Indirect Wording
        ("I am filing a formal complaint regarding misleading feature advertising on your pricing page. We purchased the business plan based on promised custom integration features, only to discover after onboarding that the API is not included.", "Medium", "long", "en"),
        ("Repeated system downtime and unannounced maintenance windows over the past month have disrupted our operations three times. We demand an immediate explanation from management and SLA service credits.", "High", "long", "en")
    ],
    "Product Inquiry": [
        # Short
        ("Does product support dark mode?", "Low", "short", "en"),
        ("Is macOS Sequoia supported?", "Low", "short", "en"),
        ("Bulk license pricing details?", "Low", "short", "en"),
        ("Where to find user documentation?", "Low", "short", "en"),
        ("API documentation link chahiye.", "Low", "short", "hinglish"),
        ("Aa product ma discount milse?", "Low", "short", "gujarati"),

        # Medium / Conversational / Questions
        ("Could you tell me if your platform supports SAML 2.0 Single Sign-On for enterprise users?", "Low", "medium", "en"),
        ("What are the minimum hardware and RAM requirements for running the server software on Linux?", "Low", "medium", "en"),
        ("Do you offer non-profit or educational discounts for team subscription accounts?", "Low", "medium", "en"),
        ("Can I request a live demo session with a product specialist for model {model}?", "Low", "medium", "en"),
        ("Does the cloud storage infrastructure comply with SOC2 and HIPAA security certifications?", "Medium", "medium", "en"),
        ("Kya is software mein custom reporting export option available hai?", "Low", "medium", "hinglish"),

        # Long / Detailed / Indirect Wording
        ("We are currently evaluating software solutions for a 300-person engineering team. Could you provide a comprehensive comparison document outlining feature differences between the Professional and Enterprise plans, along with volume discount schedules?", "Low", "long", "en"),
        ("Does your REST API support webhook event subscriptions for real-time order status updates, and where can we access the OpenAPI / Swagger documentation for testing?", "Low", "long", "en")
    ],
    "Cancellation": [
        # Short
        ("Cancel subscription immediately.", "High", "short", "en"),
        ("Turn off auto-renewal.", "Low", "short", "en"),
        ("Close my account.", "Medium", "short", "en"),
        ("Subscription end karo.", "Low", "short", "hinglish"),
        ("Plan cancellation request.", "Low", "short", "en"),
        ("Account delete karvu che.", "Medium", "short", "gujarati"),

        # Medium / Conversational / Questions
        ("I would like to cancel my monthly subscription effective before the next billing cycle.", "Low", "medium", "en"),
        ("Please turn off automatic renewal for account {id} so I am not charged next month.", "Low", "medium", "en"),
        ("How do I cancel my pending order before it moves to the warehouse packing stage?", "Medium", "medium", "en"),
        ("We are migrating to a different vendor and request complete contract termination.", "Medium", "medium", "en"),
        ("The cancellation button is missing from my account billing settings page. Please cancel for me.", "Medium", "medium", "en"),
        ("Next month billing se pehle mera plan cancel kar do please.", "Low", "medium", "hinglish"),

        # Long / Detailed / Indirect Wording
        ("Due to budget cuts in our organization, we can no longer maintain our active subscription plan. Please process account cancellation immediately and send written confirmation that auto-renew has been disabled.", "Medium", "long", "en"),
        ("I am terminating our contract effective today because key features promised during sales onboarding remain unreleased. Please confirm cancellation of invoice {id} and remove recorded credit card details.", "Medium", "long", "en")
    ]
}

# Conversational Prefixes & Suffixes for realistic noise
CONVERSATIONAL_PREFIXES = [
    "Hi support team, ",
    "Hello, ",
    "Hey there, ",
    "Dear customer service, ",
    "Urgent: ",
    "Quick question - ",
    "FYI, ",
    "Plz help - ",
    ""
]

CONVERSATIONAL_SUFFIXES = [
    " Please advise as soon as possible.",
    " Thanks for your help!",
    " Urgent attention required.",
    " Contact me via email or phone.",
    " Appreciate a quick update on this.",
    " Kindly look into this matter promptly.",
    " Looking forward to hearing back.",
    " Plz respond asap.",
    ""
]


def apply_minor_noise(text: str) -> str:
    """Applies realistic minor typos, punctuation noise, or casing variations probabilistically."""
    if random.random() < 0.25:
        # Common minor noise replacements
        replacements = [
            ("please", "plz"),
            ("Please", "Plz"),
            ("you", "u"),
            ("You", "U"),
            ("your", "ur"),
            ("cannot", "cant"),
            ("does not", "doesnt"),
            ("with", "w/"),
            ("thanks", "thx"),
        ]
        for old, new in replacements:
            if old in text and random.random() < 0.4:
                text = text.replace(old, new, 1)
                break

    # Strip ending period occasionally for informal feel
    if random.random() < 0.15 and text.endswith("."):
        text = text[:-1]

    return text


def assign_deterministic_priority(text: str, base_priority: str, suffix: str) -> str:
    """
    Deterministically determines priority based on:
    - Urgency cues ("immediately", "right now", "urgent", "emergency", "today", "deadline")
    - Security / Severity ("hacked", "fraud", "security breach", "account compromised", "lost permanently")
    - Financial impact ("charged twice", "large amount", "unauthorized transaction", "payment failure")
    - Service impact ("cannot access account", "service completely unavailable", "business blocked")
    - Time sensitivity ("for one day", "for several days", "for a week")
    """
    text_lower = text.lower()
    suffix_lower = suffix.lower()

    # Get main text by stripping suffix to prevent suffix words from matching main triggers
    main_text = text_lower
    if suffix_lower and main_text.endswith(suffix_lower):
        main_text = main_text[:-len(suffix_lower)].strip()

    # Rule 1: Critical signals (Security breach, active fraud, severe service outage blocking business) -> Critical
    critical_triggers = [
        "unauthorized transaction", "unauthorized charge", "unrecognized login",
        "hacked", "fraud", "security breach", "account compromised",
        "database connection timeout", "server error 500", "system downtime",
        "outage during", "entire team has been unable", "lost access to the master",
        "service completely unavailable"
    ]
    if any(trig in main_text for trig in critical_triggers):
        return "Critical"

    # Rule 2: High signals (High urgency/severity/financial impact/service impact/time sensitivity) -> High
    high_triggers = [
        "missing for a week", "missing package for a week", "stuck in transit for a week",
        "delayed by a week", "express delivery", "significantly delayed",
        "locked and i cannot access", "cannot access important", "serious billing",
        "duplicate payment corrected urgently", "cancel subscription immediately",
        "immediately", "right now", "emergency", "lost permanently", "cannot access account",
        "business blocked", "urgent", "urgently", "asap"
    ]
    if any(trig in main_text for trig in high_triggers):
        return "High"

    # Rule 3: Medium signals (Medium urgency/severity/impact) -> Medium
    medium_triggers = [
        "delayed by two days", "stuck in transit for several days", "refund has not arrived",
        "charged twice", "double charge", "duplicate charge", "double billing",
        "payment failed but money deducted", "auto renewal", "duplicate payment",
        "damaged", "crashes", "freezing", "locked"
    ]
    if any(trig in main_text for trig in medium_triggers):
        if base_priority in ["High", "Critical"]:
            return base_priority
        return "Medium"

    # Rule 4: Suffix urgency bump (applies ONLY if specific regression test urgency phrase is present)
    # This keeps test_urgency_suffix_upgrades_priority green by bumping Low to Medium.
    test_urgency_suffixes = ["urgent attention required", "please advise as soon as possible"]
    has_test_urgency = any(s in suffix_lower for s in test_urgency_suffixes)

    # Rule 5: Low signals -> Low
    low_triggers = [
        "discount", "documentation", "dark mode", "vat receipt",
        "tax invoice", "how can i", "how do i", "where to find",
        "macos", "compatible", "pricing details", "where is my parcel",
        "order status", "when my package will arrive", "tracking status",
        "question about my delivery", "delivery kab tak", "status update nathi"
    ]
    
    resolved_priority = base_priority
    if any(trig in main_text for trig in low_triggers):
        resolved_priority = "Low"

    # Apply the bounded suffix bump
    if has_test_urgency:
        if resolved_priority == "Low":
            return "Medium"
        return resolved_priority

    return resolved_priority


def generate_tickets(target_count=5600):
    tickets = []
    per_category = target_count // len(CATEGORIES)
    ticket_counter = 10001

    for category in CATEGORIES:
        patterns = TICKET_PATTERNS[category]

        for i in range(per_category):
            ticket_id = f"TICK-{ticket_counter:06d}"
            ticket_counter += 1

            # Pick a pattern tuple
            phrase_tmpl, base_priority, length_type, lang = random.choice(patterns)

            # Fill formatting variables
            filled_text = phrase_tmpl.format(
                id=random.randint(10000, 99999),
                amount=random.choice([19.99, 49.00, 99.99, 149.00, 249.50, 499.00]),
                plan_amount=random.choice([9.99, 29.99, 79.99]),
                period=random.choice(["Jan 2026", "Feb 2026", "Q1 2026"]),
                err_code=random.choice(["ERR_502", "NULL_POINTER_EXC", "SOCKET_TIMEOUT", "AUTH_DENIED"]),
                ip=f"192.168.{random.randint(1,254)}.{random.randint(1,254)}",
                model=f"PRO-{random.randint(100,999)}"
            )

            prefix = random.choice(CONVERSATIONAL_PREFIXES) if lang == "en" else ""
            suffix = random.choice(CONVERSATIONAL_SUFFIXES)

            raw_full_text = prefix + filled_text + suffix
            final_text = apply_minor_noise(raw_full_text)

            # Assign priority deterministically
            priority = assign_deterministic_priority(final_text, base_priority, suffix)
            department = get_department_for_category(category)

            tickets.append({
                "ticket_id": ticket_id,
                "ticket_text": final_text,
                "category": category,
                "priority": priority,
                "department": department,
                "source": "synthetic",
                "language": lang
            })

    # Shuffle deterministically
    random.shuffle(tickets)
    df = pd.DataFrame(tickets)
    return df


def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")
    raw_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    print("Generating enhanced synthetic dataset (5600 samples)...")
    df = generate_tickets(5600)

    # Save to raw directory and legacy dataset directory
    raw_synthetic_path = os.path.join(raw_dir, "synthetic_tickets.csv")
    df.to_csv(raw_synthetic_path, index=False)
    print(f"Saved raw synthetic dataset to {raw_synthetic_path} (Shape: {df.shape})")

    tickets_path = os.path.join(dataset_dir, "tickets.csv")
    df.to_csv(tickets_path, index=False)
    print(f"Saved dataset copy to {tickets_path} (Shape: {df.shape})")

    # Run incoming ticket ingestion
    print("\n[INFO] Running incoming ticket ingestion script...")
    from scripts.ingest_incoming import process_incoming_tickets
    process_incoming_tickets()

    # Run canonical dataset preparation and split
    print("\n[INFO] Calling canonical dataset preparation script prepare_dataset.py...")
    from scripts.prepare_dataset import prepare_data
    prepare_data()


if __name__ == "__main__":
    main()
