FINANCE_DISCLAIMER = (
    "This assistant provides educational information only and should not be treated as financial, tax, or investment advice. "
    "Consult a qualified professional before making decisions."
)


def attach_disclaimer(message: str) -> str:
    return f"{message}\n\n⚠️ {FINANCE_DISCLAIMER}"
