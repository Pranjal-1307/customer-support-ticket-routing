import pandas as pd
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_dir = os.path.join(base_dir, "data", "raw")
eval_dir = os.path.join(base_dir, "data", "evaluation")
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)

# 1. MANUAL TICKETS (Diverse tone, style, typos, length)
manual_data = [
    # Billing
    ("MAN_B001", "My bank statement indicates that the same transaction was processed twice.", "Billing", "High", "manual"),
    ("MAN_B002", "Why did u guys charge me again??", "Billing", "High", "manual"),
    ("MAN_B003", "This is ridiculous, I have been charged twice and nobody is helping me!", "Billing", "Critical", "manual"),
    ("MAN_B004", "Charged twice.", "Billing", "High", "manual"),
    ("MAN_B005", "I purchased the annual subscription yesterday, but my credit card statement now shows two separate charges for the same order.", "Billing", "High", "manual"),
    ("MAN_B006", "Need vat invoice for last month subscription fee.", "Billing", "Low", "manual"),
    ("MAN_B007", "Auto renewal billed me $99 even though I sent cancellation request last week.", "Billing", "High", "manual"),
    ("MAN_B008", "Incorrect tax amount added to invoice #INV-99201. Plz fix.", "Billing", "Medium", "manual"),
    ("MAN_B009", "Payment went through on my bank app but portal says payment pending.", "Billing", "Medium", "manual"),
    ("MAN_B100", "Can you send the PDF receipt for my team subscription to accounting@company.org?", "Billing", "Low", "manual"),
    ("MAN_B101", "Overcharged by 50 dollars on month end billing statement.", "Billing", "High", "manual"),
    ("MAN_B102", "I was charged currency conversion fee when purchase was in USD. Why?", "Billing", "Medium", "manual"),
    ("MAN_B103", "Promo code PROMO2024 was applied but full price was charged to my card.", "Billing", "Medium", "manual"),
    ("MAN_B104", "Got a message saying payment failed, but my account balance decreased.", "Billing", "High", "manual"),
    ("MAN_B105", "Why is there an additional $15 maintenance charge on my bill?", "Billing", "Medium", "manual"),

    # Technical Support
    ("MAN_T001", "The app crashes whenever I try to upload a file.", "Technical Support", "High", "manual"),
    ("MAN_T002", "Getting 500 error when clicking submit.", "Technical Support", "Medium", "manual"),
    ("MAN_T003", "Hey, your website is completely down on Chrome. White screen showing up.", "Technical Support", "Critical", "manual"),
    ("MAN_T004", "API response time is over 12 seconds for /v1/data endpoint. Server timeout.", "Technical Support", "High", "manual"),
    ("MAN_T005", "Cant connect database error sqlite3.OperationalError database locked.", "Technical Support", "High", "manual"),
    ("MAN_T006", "Export to CSV button does nothing when clicked in Firefox browser.", "Technical Support", "Medium", "manual"),
    ("MAN_T007", "App stuck on loading screen after latest 2.4.1 update.", "Technical Support", "High", "manual"),
    ("MAN_T008", "Dashboard metrics showing 0 despite active data ingestion streams.", "Technical Support", "Medium", "manual"),
    ("MAN_T009", "SSL certificate expired warning when accessing client portal.", "Technical Support", "Critical", "manual"),
    ("MAN_T100", "Mobile app logout spontaneously every 5 minutes.", "Technical Support", "Medium", "manual"),
    ("MAN_T101", "Dark mode toggle is glitching and flickering constantly.", "Technical Support", "Low", "manual"),
    ("MAN_T102", "Desktop software crash dump attached - exit code 0xC0000005.", "Technical Support", "High", "manual"),
    ("MAN_T103", "Unable to sync offline offline records to server once online.", "Technical Support", "High", "manual"),
    ("MAN_T104", "Notification push messages are not arriving on Android 14 devices.", "Technical Support", "Medium", "manual"),

    # Refund
    ("MAN_R001", "Item arrived damaged in transit. I want an immediate full refund.", "Refund", "High", "manual"),
    ("MAN_R002", "I requested refund 10 days ago. Still no update on status???", "Refund", "High", "manual"),
    ("MAN_R003", "Product did not match the website description at all. Returning product for money back.", "Refund", "Medium", "manual"),
    ("MAN_R004", "Accidentally bought 2 licenses instead of 1. Refund the extra one please.", "Refund", "Medium", "manual"),
    ("MAN_R005", "Order cancelled by seller due to out of stock, but money not refunded yet.", "Refund", "High", "manual"),
    ("MAN_R006", "Where is my refund?", "Refund", "Medium", "manual"),
    ("MAN_R007", "Partial refund received ($20) instead of promised full refund ($100).", "Refund", "High", "manual"),
    ("MAN_R008", "System charged me after trial ended even though I opted out. Money back immediately.", "Refund", "Critical", "manual"),
    ("MAN_R009", "Service quality was unacceptable during webinar event. Requesting subscription refund.", "Refund", "Medium", "manual"),
    ("MAN_R100", "Returned item received at warehouse per carrier tracking #8821. Process reimbursement.", "Refund", "Medium", "manual"),

    # Shipping
    ("MAN_S001", "Where can I track my delivery?", "Shipping", "Low", "manual"),
    ("MAN_S002", "Package marked delivered by courier but I did not receive anything at my door.", "Shipping", "Critical", "manual"),
    ("MAN_S003", "Delivery is 4 days late and tracking status has not updated since Monday.", "Shipping", "High", "manual"),
    ("MAN_S004", "Need to change shipping address to Apartment 4B before order dispatches.", "Shipping", "High", "manual"),
    ("MAN_S005", "Item shipped to old residential address by mistake.", "Shipping", "High", "manual"),
    ("MAN_S006", "Parcel box arrived completely crushed and open.", "Shipping", "High", "manual"),
    ("MAN_S007", "Paid extra for next-day priority shipping, but it's taking standard 5 days.", "Shipping", "Medium", "manual"),
    ("MAN_S008", "Customs holding shipment at port demanding additional clearance documentation.", "Shipping", "Medium", "manual"),
    ("MAN_S009", "Wrong tracking code sent in email confirmation.", "Shipping", "Low", "manual"),
    ("MAN_S100", "Courier driver left package in heavy rain without ringing doorbell.", "Shipping", "Medium", "manual"),

    # Account
    ("MAN_A001", "Locked out of my admin account due to forgotten 2FA device.", "Account", "Critical", "manual"),
    ("MAN_A002", "Password reset link is giving 'expired' error immediately upon arrival.", "Account", "High", "manual"),
    ("MAN_A003", "How do I change my primary account email address?", "Account", "Low", "manual"),
    ("MAN_A004", "Suspicious login attempt detected from unrecognized location. Deactivate account temporarily.", "Account", "Critical", "manual"),
    ("MAN_A005", "Cannot update profile picture or display name in settings page.", "Account", "Low", "manual"),
    ("MAN_A006", "Please transfer account owner privileges to user user2@org.com.", "Account", "Medium", "manual"),
    ("MAN_A007", "Verification code SMS not reaching my mobile phone number.", "Account", "High", "manual"),
    ("MAN_A008", "Account deactivated message showing when logging in. Why?", "Account", "High", "manual"),
    ("MAN_A009", "Want to delete my profile and purge all stored user personal data.", "Account", "Medium", "manual"),
    ("MAN_A100", "SSO Google login fails with OAuth redirect URI mismatch.", "Account", "High", "manual"),

    # Complaint
    ("MAN_C001", "Your customer service rep on phone was extremely rude and unhelpful.", "Complaint", "High", "manual"),
    ("MAN_C002", "Worst experience ever. System has been down 3 times this month!", "Complaint", "High", "manual"),
    ("MAN_C003", "I have been waiting on hold for over 45 minutes without any agent answering.", "Complaint", "Medium", "manual"),
    ("MAN_C004", "Misleading marketing claims regarding features included in standard plan.", "Complaint", "Medium", "manual"),
    ("MAN_C005", "Escalating this issue to management. Unresolved bug open for 30 days.", "Complaint", "Critical", "manual"),
    ("MAN_C006", "Prices increased by 30% without prior email notification to subscribers.", "Complaint", "High", "manual"),
    ("MAN_C007", "Support ticket #4401 closed automatically without providing solution.", "Complaint", "High", "manual"),
    ("MAN_C008", "Data privacy concern: shared workspace showing other client's file names.", "Complaint", "Critical", "manual"),

    # Product Inquiry
    ("MAN_P001", "Does the enterprise tier support single sign-on (SSO) via SAML 2.0?", "Product Inquiry", "Low", "manual"),
    ("MAN_P002", "Is there an iOS tablet version available for download?", "Product Inquiry", "Low", "manual"),
    ("MAN_P003", "What is the maximum file upload size limit per attachment?", "Product Inquiry", "Low", "manual"),
    ("MAN_P004", "Can I integrate this platform with Zapier or Webhooks?", "Product Inquiry", "Low", "manual"),
    ("MAN_P005", "Looking for detailed documentation on REST API rate limits.", "Product Inquiry", "Low", "manual"),
    ("MAN_P006", "Are there discount options for non-profit organizations or universities?", "Product Inquiry", "Low", "manual"),
    ("MAN_P007", "Does product support dark theme and custom CSS styling?", "Product Inquiry", "Low", "manual"),
    ("MAN_P008", "Is user training included with team plan onboard?", "Product Inquiry", "Low", "manual"),

    # Cancellation
    ("MAN_X001", "Please cancel my subscription immediately before next billing cycle.", "Cancellation", "High", "manual"),
    ("MAN_X002", "How do I terminate my account subscription?", "Cancellation", "Medium", "manual"),
    ("MAN_X003", "Cancel auto-renewal for my annual membership plan.", "Cancellation", "Medium", "manual"),
    ("MAN_X004", "Want to end contract early due to company restructuring.", "Cancellation", "High", "manual"),
    ("MAN_X005", "Stop my subscription at end of current billing period.", "Cancellation", "Medium", "manual"),
    ("MAN_X006", "I am closing my business and need to cancel all associated services.", "Cancellation", "High", "manual"),
    ("MAN_X007", "Unsubscribe me from all paid services and emails right now.", "Cancellation", "High", "manual"),
    ("MAN_X008", "Switched to alternative product, please cancel plan effective today.", "Cancellation", "Medium", "manual"),
]

df_manual = pd.DataFrame(manual_data, columns=["ticket_id", "ticket_text", "category", "priority", "source"])
df_manual.to_csv(os.path.join(raw_dir, "manual_tickets.csv"), index=False)
print(f"Created manual_tickets.csv: {len(df_manual)} rows")


# 2. EXTERNAL TICKETS (Realistic public-style customer support dataset)
external_data = [
    # Billing
    ("EXT_B001", "Received billing notification for unknown monthly service charge.", "Billing", "High", "external"),
    ("EXT_B002", "Updated my payment method but automated billing still shows expired card.", "Billing", "Medium", "external"),
    ("EXT_B003", "Requesting refund of duplicate charge on transaction statement #9928.", "Billing", "High", "external"),
    ("EXT_B004", "Invoice generated without our corporate tax identification number.", "Billing", "Low", "external"),
    ("EXT_B005", "Credit card failed error during checkout process.", "Billing", "High", "external"),
    ("EXT_B006", "Billed standard rate instead of contracted enterprise discount rate.", "Billing", "High", "external"),
    ("EXT_B007", "Need billing contact updated for accounts payable department.", "Billing", "Low", "external"),
    ("EXT_B008", "Annual invoice shows wrong seat count for active user licenses.", "Billing", "Medium", "external"),
    ("EXT_B009", "Payment confirmation email not received after successful wire transfer.", "Billing", "Low", "external"),
    ("EXT_B010", "Discrepancy between quote provided by sales rep and billed invoice total.", "Billing", "High", "external"),

    # Technical Support
    ("EXT_T001", "Database connector throwing connection reset by peer error.", "Technical Support", "Critical", "external"),
    ("EXT_T002", "User authentication token expiring prematurely causing frequent re-logins.", "Technical Support", "Medium", "external"),
    ("EXT_T003", "Web portal rendering blank white page on Safari browser version 17.", "Technical Support", "High", "external"),
    ("EXT_T004", "File conversion utility hanging at 99% progress bar indefinitely.", "Technical Support", "Medium", "external"),
    ("EXT_T005", "High latency reported on regional server node us-east-1.", "Technical Support", "High", "external"),
    ("EXT_T006", "Memory leak observed in desktop daemon process background worker.", "Technical Support", "High", "external"),
    ("EXT_T007", "Webhook endpoint returning HTTP 403 Forbidden on payload POST.", "Technical Support", "High", "external"),
    ("EXT_T008", "Search indexing engine failed to update new product listings.", "Technical Support", "Medium", "external"),
    ("EXT_T009", "Client side validation error preventing form submission on checkout.", "Technical Support", "Medium", "external"),
    ("EXT_T010", "System audit log export truncating records older than 30 days.", "Technical Support", "Low", "external"),

    # Refund
    ("EXT_R001", "Requesting full reimbursement for cancelled conference ticket.", "Refund", "Medium", "external"),
    ("EXT_R002", "Returned goods delivered to facility on 12th, awaiting credit posting.", "Refund", "Medium", "external"),
    ("EXT_R003", "Product hardware malfunctioning within 14-day return window.", "Refund", "High", "external"),
    ("EXT_R004", "Overcharge refund agreed by support representative not reflected on card.", "Refund", "High", "external"),
    ("EXT_R005", "Cancelled order refund status inquiry for order reference #77291.", "Refund", "Medium", "external"),
    ("EXT_R006", "Subscription cancelled during money-back guarantee period.", "Refund", "High", "external"),
    ("EXT_R007", "Defective shipment returned, please credit back original payment source.", "Refund", "High", "external"),
    ("EXT_R008", "Service SLA breach credit refund request for downtime incident.", "Refund", "High", "external"),
    ("EXT_R009", "Double payment deduction refund for transaction ID #10492.", "Refund", "Critical", "external"),
    ("EXT_R010", "Order items damaged during transport, demanding store credit or refund.", "Refund", "High", "external"),

    # Shipping
    ("EXT_S001", "Carrier notification indicates delivery delay due to severe weather conditions.", "Shipping", "Low", "external"),
    ("EXT_S002", "Package tracking details not updated for past 72 hours.", "Shipping", "Medium", "external"),
    ("EXT_S003", "Shipment delivered to wrong zip code according to tracking link.", "Shipping", "High", "external"),
    ("EXT_S004", "Request to hold parcel at local depot for self pickup.", "Shipping", "Low", "external"),
    ("EXT_S005", "International freight shipment stuck in customs inspection.", "Shipping", "Medium", "external"),
    ("EXT_S006", "Dispatch confirmation received but shipping label show pending pickup.", "Shipping", "Low", "external"),
    ("EXT_S007", "Partial order shipment arrived, remaining items missing from package.", "Shipping", "High", "external"),
    ("EXT_S008", "Address correction required prior to courier dispatch.", "Shipping", "High", "external"),
    ("EXT_S009", "Damaged packaging container reported upon freight arrival.", "Shipping", "Medium", "external"),
    ("EXT_S010", "Express delivery deadline missed by courier service provider.", "Shipping", "Medium", "external"),

    # Account
    ("EXT_A001", "Multi-factor authentication reset required for admin profile.", "Account", "Critical", "external"),
    ("EXT_A002", "Unable to modify account owner email in security settings panel.", "Account", "Medium", "external"),
    ("EXT_A003", "Account locked after automated security audit flagged anomaly.", "Account", "High", "external"),
    ("EXT_A004", "Password reset verification email not arriving in inbox.", "Account", "High", "external"),
    ("EXT_A005", "User permission escalation request for team manager profile.", "Account", "Medium", "external"),
    ("EXT_A006", "Deactivate former employee user access from corporate domain.", "Account", "High", "external"),
    ("EXT_A007", "Single Sign-On integration failing with identity provider metadata error.", "Account", "High", "external"),
    ("EXT_A008", "Profile contact phone number update request.", "Account", "Low", "external"),
    ("EXT_A009", "GDPR account deletion request and data erasure.", "Account", "Medium", "external"),
    ("EXT_A010", "Concurrent session limit error blocking valid user login.", "Account", "Medium", "external"),

    # Complaint
    ("EXT_C001", "Unacceptable response time from Tier 1 support team.", "Complaint", "High", "external"),
    ("EXT_C002", "Repeated platform outages impacting critical business operations.", "Complaint", "Critical", "external"),
    ("EXT_C003", "Feature promised during sales call is absent from released product.", "Complaint", "High", "external"),
    ("EXT_C004", "Unresolved software defect open for over six weeks.", "Complaint", "High", "external"),
    ("EXT_C005", "Poor communication regarding planned system maintenance window.", "Complaint", "Medium", "external"),
    ("EXT_C006", "Unexpected price structure changes without prior advisory notice.", "Complaint", "High", "external"),
    ("EXT_C007", "Support ticket marked resolved without customer confirmation.", "Complaint", "High", "external"),
    ("EXT_C008", "Inconsistent information provided by different support reps.", "Complaint", "Medium", "external"),

    # Product Inquiry
    ("EXT_P001", "Inquiring about compatibility with PostgreSQL 15 database engines.", "Product Inquiry", "Low", "external"),
    ("EXT_P002", "What encryption standards are utilized for data at rest?", "Product Inquiry", "Low", "external"),
    ("EXT_P003", "Are custom API rate limits available for enterprise contracts?", "Product Inquiry", "Low", "external"),
    ("EXT_P004", "Does the system support multi-currency transactions and reporting?", "Product Inquiry", "Low", "external"),
    ("EXT_P005", "Inquiry regarding roadmap for native mobile tablet application.", "Product Inquiry", "Low", "external"),
    ("EXT_P006", "What is the maximum number of concurrent users supported on Pro plan?", "Product Inquiry", "Low", "external"),
    ("EXT_P007", "Requesting hardware specification guide for on-premise installation.", "Product Inquiry", "Low", "external"),
    ("EXT_P008", "Is SOC2 Type II compliance documentation available upon NDA?", "Product Inquiry", "Low", "external"),

    # Cancellation
    ("EXT_X001", "Formal notice of contract non-renewal effective at term end.", "Cancellation", "High", "external"),
    ("EXT_X002", "Request to cancel automatic monthly plan renewal.", "Cancellation", "Medium", "external"),
    ("EXT_X003", "Immediate account termination and billing cancellation requested.", "Cancellation", "High", "external"),
    ("EXT_X004", "Subscription cancellation due to business acquisition.", "Cancellation", "Medium", "external"),
    ("EXT_X005", "Downgrading from Premium to Free plan before billing cycle.", "Cancellation", "Medium", "external"),
    ("EXT_X006", "Terminate trial period and remove credit card details.", "Cancellation", "Medium", "external"),
    ("EXT_X007", "Cancelling membership due to budget constraints.", "Cancellation", "Medium", "external"),
    ("EXT_X008", "Account closure request following vendor consolidation.", "Cancellation", "High", "external"),
]

df_ext = pd.DataFrame(external_data, columns=["ticket_id", "ticket_text", "category", "priority", "source"])
df_ext.to_csv(os.path.join(raw_dir, "external_tickets.csv"), index=False)
print(f"Created external_tickets.csv: {len(df_ext)} rows")


# 3. UNSEEN EVALUATION TEST SET
unseen_data = [
    ("UNSEEN_001", "I can see two identical transactions on my card for one purchase.", "Billing", "High", "unseen_eval"),
    ("UNSEEN_002", "The application shuts down immediately after I enter my password.", "Technical Support", "High", "unseen_eval"),
    ("UNSEEN_003", "I cancelled my plan last week, but another renewal payment has been taken.", "Billing", "Critical", "unseen_eval"),
    ("UNSEEN_004", "Our entire office has lost access to the service.", "Technical Support", "Critical", "unseen_eval"),
    ("UNSEEN_005", "My parcel was supposed to arrive last Friday and there is still no update.", "Shipping", "High", "unseen_eval"),
    ("UNSEEN_006", "I need help understanding why an extra amount was added to my invoice.", "Billing", "Medium", "unseen_eval"),
    ("UNSEEN_007", "Sent the returned item back 3 weeks ago, when is my cash returning?", "Refund", "High", "unseen_eval"),
    ("UNSEEN_008", "I want to shut down my company account forever.", "Cancellation", "High", "unseen_eval"),
    ("UNSEEN_009", "Is your platform compliant with HIPAA regulations for medical data?", "Product Inquiry", "Low", "unseen_eval"),
    ("UNSEEN_010", "Your representative hung up on me while I was explaining my problem!", "Complaint", "Critical", "unseen_eval"),
    ("UNSEEN_011", "Forgot my master password and backup keys are not working.", "Account", "Critical", "unseen_eval"),
    ("UNSEEN_012", "Courier driver left box outside in torrential rain storm.", "Shipping", "High", "unseen_eval"),
    ("UNSEEN_013", "Web portal throws 404 page not found when accessing user profile.", "Technical Support", "Medium", "unseen_eval"),
    ("UNSEEN_014", "Can I upgrade from monthly to annual billing and get a discount?", "Billing", "Low", "unseen_eval"),
    ("UNSEEN_015", "Product arrived with shattered glass casing.", "Refund", "High", "unseen_eval"),
]

df_unseen = pd.DataFrame(unseen_data, columns=["ticket_id", "ticket_text", "category", "priority", "source"])
df_unseen.to_csv(os.path.join(eval_dir, "unseen_test.csv"), index=False)
print(f"Created unseen_test.csv: {len(df_unseen)} rows")


# 4. HINGLISH EVALUATION TEST SET
hinglish_data = [
    ("HING_001", "mera payment do baar deduct ho gaya", "Billing", "High", "hinglish_eval"),
    ("HING_002", "app login nahi ho raha hai error aa raha hai", "Technical Support", "High", "hinglish_eval"),
    ("HING_003", "order abhi tak deliver nahi hua kitna time lagega", "Shipping", "Medium", "hinglish_eval"),
    ("HING_004", "refund kab milega do hafte ho gaye", "Refund", "High", "hinglish_eval"),
    ("HING_005", "password reset link email pe nahi aaya", "Account", "Medium", "hinglish_eval"),
    ("HING_006", "subscription cancel karna hai please help karo", "Cancellation", "Medium", "hinglish_eval"),
    ("HING_007", "ye product me sso support hai kya?", "Product Inquiry", "Low", "hinglish_eval"),
    ("HING_008", "customer care wale koi phone nahi utha rahe", "Complaint", "High", "hinglish_eval"),
    ("HING_009", "bank statement me extra charge kyu dikha raha hai", "Billing", "High", "hinglish_eval"),
    ("HING_100", "courier wala parcel galat jagah deke chala gaya", "Shipping", "Critical", "hinglish_eval"),
]

df_hinglish = pd.DataFrame(hinglish_data, columns=["ticket_id", "ticket_text", "category", "priority", "source"])
df_hinglish.to_csv(os.path.join(eval_dir, "hinglish_test.csv"), index=False)
print(f"Created hinglish_test.csv: {len(df_hinglish)} rows")
