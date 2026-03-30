"""
utils.py  –  Formatting helpers
"""

def format_currency(value: float | int) -> str:
    if not value or value == 0:
        return "N/A"
    if value >= 1e12:
        return f"${value/1e12:.2f}T"
    if value >= 1e9:
        return f"${value/1e9:.2f}B"
    if value >= 1e6:
        return f"${value/1e6:.2f}M"
    return f"${value:,.0f}"


def format_percent(value: float) -> str:
    return f"{value:+.2f}%"


def color_metric(value: float, good_is_positive: bool = True) -> str:
    """Return HTML-coloured metric string."""
    if good_is_positive:
        clr = "#22c55e" if value >= 0 else "#ef4444"
    else:
        clr = "#ef4444" if value >= 0 else "#22c55e"
    return f'<span style="color:{clr};">{value:+.2f}%</span>'
