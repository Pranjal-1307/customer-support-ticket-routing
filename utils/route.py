"""
Routing Engine Module.
Maps predicted category to the appropriate target department/support team.
"""

CATEGORY_DEPARTMENT_MAP = {
    "Billing": "Billing Team",
    "Technical Support": "Technical Support Team",
    "Refund": "Refund Team",
    "Shipping": "Logistics Team",
    "Account": "Account Support Team",
    "Complaint": "Customer Care Team",
    "Product Inquiry": "Product Support Team",
    "Cancellation": "Retention Team"
}


def get_department_for_category(category: str) -> str:
    """
    Returns assigned department name given predicted category.
    Defaults to 'General Support Team' if category is unmapped.
    """
    if not category:
        return "General Support Team"
    return CATEGORY_DEPARTMENT_MAP.get(category.strip(), "General Support Team")
