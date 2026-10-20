"""Conservative validation for customer-submitted ticket descriptions."""

import re
from dataclasses import dataclass
from typing import Optional


MAX_TICKET_LENGTH = 5000
"""Maximum accepted ticket description length, in characters."""

_KEYBOARD_MASHES = {
    "asdf",
    "asdfghjkl",
    "qwertyuiop",
    "zxcvbnm",
    "poiuytrewq",
    "lkjhgfdsa",
    "mnbvcxz",
}


@dataclass(frozen=True)
class TicketTextValidation:
    ticket_text: Optional[str]
    message: Optional[str] = None
    status_code: int = 400

    @property
    def is_valid(self) -> bool:
        return self.ticket_text is not None


def validate_ticket_text(value: object) -> TicketTextValidation:
    """Trim and conservatively reject empty, non-descriptive, or obvious gibberish input."""
    if not isinstance(value, str):
        return TicketTextValidation(None, "Please enter a ticket description.")

    if len(value) > MAX_TICKET_LENGTH:
        return TicketTextValidation(
            None,
            f"Please keep your ticket description under {MAX_TICKET_LENGTH} characters.",
            413,
        )

    ticket_text = value.strip()
    if not ticket_text:
        return TicketTextValidation(None, "Please enter a ticket description.")

    # Letters are required so numbers, punctuation, symbols, and emoji alone are rejected.
    if not any(character.isalpha() for character in ticket_text):
        return TicketTextValidation(None, "Please describe the issue using words.")

    tokens = re.findall(r"[^\W_]+", ticket_text, flags=re.UNICODE)
    if any(len(token) >= 6 and len(set(token)) == 1 for token in tokens):
        return TicketTextValidation(None, "Please enter a meaningful description of the issue.")

    mash_tokens = [token.casefold() for token in tokens if token.casefold() in _KEYBOARD_MASHES]
    descriptive_tokens = [
        token for token in tokens
        if token.isalpha() and token.casefold() not in _KEYBOARD_MASHES
    ]
    # Permit a keyboard sequence mentioned within an otherwise descriptive sentence,
    # while rejecting standalone mashes and combinations such as "asdf 1234 !!!".
    if mash_tokens and not descriptive_tokens:
        return TicketTextValidation(None, "Please enter a meaningful description of the issue.")

    return TicketTextValidation(ticket_text)
