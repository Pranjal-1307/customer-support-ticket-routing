"""
Priority Signals Feature Extraction Module for Customer Support Tickets (M3).
Extracts 6 interpretable, continuous priority signal features (0.0 to 1.0):
1. urgency: Explicit demand for immediate response/action.
2. severity: Magnitude of malfunction, error, damage, or blocking issue.
3. financial_impact: Direct financial loss, unauthorized billing, duplicate charges.
4. security_risk: Account takeover, credential compromise, data breach, fraud.
5. service_impact: System outage, total downtime, business disruption, inaccessible core features.
6. time_sensitivity: Specific deadlines, time constraints (today, 24h, ASAP, delayed weeks).
"""

import re
from typing import Dict, Any, List

# Signal Keyword & Regex Patterns (Multilingual: English, Hinglish, Gujarati)
PATTERNS = {
    "urgency": [
        (r"\b(urgent|urgently|emergency|asap|immediate|immediately|crisis|critical|priority)\b", 0.9),
        (r"\b(jaldi|turant|ekdum|fatafat|zaroori|urgent\s+hai)\b", 0.9),
        (r"(તરત|જલ્દી|તાત્કાલિક|ઇમરજન્સી)", 0.9),
        (r"\b(plz\s+respond\s+asap|respond\s+asap|need\s+help\s+now|right\s+now|at\s+once)\b", 0.85),
        (r"\b(soon\s+as\s+possible|quick\s+update|please\s+hurry|hurry\s+up)\b", 0.6),
        (r"\b(plz\s+help|quick\s+question|can\s+you\s+help|looking\s+forward)\b", 0.2),
    ],
    "severity": [
        (r"\b(completely\s+broken|crash|crashing|fatal|unusable|destroyed|severe|critical\s+bug)\b", 0.95),
        (r"\b(kharab|kaam\s+nahi\s+kar\s+raha|chal\s+nahi\s+raha|band\s+hai)\b", 0.8),
        (r"(ક્રેશ|કામ\s*નથી\s*કરતું|બંધ\s*છે|ખરાબ)", 0.8),
        (r"\b(broken|corrupted|error\s+code|500\s+internal|failed\s+completely|dead)\b", 0.75),
        (r"\b(glitch|minor\s+issue|slow|intermittent|not\s+working\s+properly)\b", 0.4),
        (r"\b(inquiry|status|how\s+to|where\s+is|what\s+is)\b", 0.1),
    ],
    "financial_impact": [
        (r"\b(charged\s+twice|double\s+charge|unauthorized\s+charge|fraudulent\s+transaction|stolen\s+card|money\s+deducted)\b", 0.95),
        (r"\b(paise\s+kat\s+gaye|do\s+baar\s+(?:charge|kata|kat|kate|payment)|paisa\s+chori|galat\s+charge|card\s+charge)\b", 0.95),
        (r"(બે\s*વાર\s*પૈસા|પૈસા\s*કપાઈ|ખોટો\s*ચાર્જ|બે\s*વાર\s*ચાર્જ)", 0.95),
        (r"\b(refund|overcharged|payment\s+failed|billing\s+error|wrong\s+amount|invoice\s+issue)\b", 0.75),
        (r"\b(paise\s+wapas|refund\s+chahiye|charge\s+back)\b", 0.75),
        (r"(રિફંડ|પૈસા\s*પાછા)", 0.75),
        (r"\b(pricing|subscription|cost|receipt|estimate)\b", 0.3),
    ],
    "security_risk": [
        (r"\b(hacked|account\s+hacked|unauthorized\s+access|breach|compromised|data\s+leak)\b", 1.0),
        (r"\b(hack\s+ho\s+gaya|chori\s+ho\s+gaya|account\s+kisi\s+aur\s+ke\s+paas)\b", 1.0),
        (r"(હેક|હેક\s*થઈ|ચોરાઈ|એકાઉન્ટ\s*હેક)", 1.0),
        (r"\b(suspicious\s+activity|password\s+reset\s+without|phishing|stolen\s+credentials|impersonation)\b", 0.85),
        (r"\b(cannot\s+login|locked\s+out|blocked\s+account|2fa\s+failed|otp\s+issue)\b", 0.6),
        (r"\b(update\s+profile|change\s+email|update\s+password|profile\s+information)\b", 0.2),
    ],
    "service_impact": [
        (r"\b(entire\s+business|all\s+users|production\s+down|system\s+outage|total\s+downtime|enterprise\s+down)\b", 1.0),
        (r"\b(poora\s+system\s+down|sabka\s+band|server\s+down)\b", 0.9),
        (r"(આખી\s*સિસ્ટમ\s*ડાઉન|સર્વર\s*ડાઉન|બધું\s*બંધ)", 0.9),
        (r"\b(server\s+error|service\s+unavailable|api\s+down|database\s+connection\s+failed|gateway\s+timeout)\b", 0.8),
        (r"\b(single\s+feature|page\s+not\s+loading|unable\s+to\s+checkout|button\s+not\s+working)\b", 0.5),
        (r"\b(cosmetic|spelling|styling|general\s+feedback)\b", 0.1),
    ],
    "time_sensitivity": [
        (r"\b(today|within\s+an\s+hour|24\s+hours|deadline|immediately|by\s+tonight)\b", 0.9),
        (r"\b(aaj\s+hi|aaj\s+shaam\s+tak|abhi\s+ke\s+abhi|jaldi\s+se\s+jaldi)\b", 0.9),
        (r"(આજે\s*જ|તરત\s*જ|૨૪\s*કલાક)", 0.9),
        (r"\b(delayed\s+for\s+(?:two|several|\d+)\s+weeks|waiting\s+for\s+(?:weeks|months)|long\s+overdue)\b", 0.85),
        (r"\b(delayed\s+by\s+\d+\s+days|late\s+by\s+\d+\s+days|two\s+days\s+late|overdue)\b", 0.65),
        (r"\b(this\s+week|in\s+coming\s+days|no\s+rush|whenever\s+possible)\b", 0.2),
    ]
}


def extract_priority_signals(ticket_text: str) -> Dict[str, float]:
    """
    Analyzes ticket text and extracts 6 priority signal scores between 0.0 and 1.0.
    Returns structured dictionary with continuous signal scores.
    """
    if not isinstance(ticket_text, str) or not ticket_text.strip():
        return {
            "urgency": 0.0,
            "severity": 0.0,
            "financial_impact": 0.0,
            "security_risk": 0.0,
            "service_impact": 0.0,
            "time_sensitivity": 0.0
        }

    text_lower = ticket_text.lower()
    signals = {}

    for signal_name, rules in PATTERNS.items():
        max_score = 0.0
        for pattern, score in rules:
            if re.search(pattern, text_lower):
                if score > max_score:
                    max_score = score
        signals[signal_name] = round(max_score, 3)

    return signals


def signals_to_feature_vector(signals: Dict[str, float]) -> List[float]:
    """Converts signals dictionary into ordered list of floats for ML vectorization."""
    ordered_keys = ["urgency", "severity", "financial_impact", "security_risk", "service_impact", "time_sensitivity"]
    return [float(signals.get(k, 0.0)) for k in ordered_keys]


def generate_priority_explanation(signals: Dict[str, float], predicted_priority: str = "Medium") -> str:
    """
    Generates human-readable, domain-interpretable reasoning for the priority prediction.
    """
    reasons = []
    
    if signals.get("security_risk", 0.0) >= 0.7:
        reasons.append("critical security/credential risk")
    elif signals.get("security_risk", 0.0) >= 0.4:
        reasons.append("account access concern")

    if signals.get("service_impact", 0.0) >= 0.7:
        reasons.append("major system/service disruption")
    elif signals.get("service_impact", 0.0) >= 0.4:
        reasons.append("service functionality limitation")

    if signals.get("financial_impact", 0.0) >= 0.8:
        reasons.append("significant financial/unauthorized charge impact")
    elif signals.get("financial_impact", 0.0) >= 0.5:
        reasons.append("billing/refund inquiry")

    if signals.get("urgency", 0.0) >= 0.7:
        reasons.append("high urgency demand")

    if signals.get("time_sensitivity", 0.0) >= 0.7:
        reasons.append("strict time sensitivity or severe delay")
    elif signals.get("time_sensitivity", 0.0) >= 0.5:
        reasons.append("delivery delay")

    if signals.get("severity", 0.0) >= 0.7:
        reasons.append("severe system or component failure")

    if not reasons:
        if predicted_priority in ["Low", "Medium"]:
            return f"{predicted_priority} priority: Standard routine inquiry with no elevated urgency or risk signals."
        else:
            return f"{predicted_priority} priority: Moderately elevated priority based on ticket context."

    reason_str = ", ".join(reasons)
    return f"{predicted_priority} priority: {reason_str.capitalize()} detected."
