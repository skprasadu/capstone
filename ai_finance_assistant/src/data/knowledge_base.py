from dataclasses import dataclass
from typing import List


@dataclass
class Article:
    title: str
    category: str
    url: str
    content: str  # NEW


def seed_articles() -> List[Article]:
    return [
        Article(
            title="Why diversification matters",
            category="portfolio_basics",
            url="https://example.com/diversification",
            content=(
                "Diversification means spreading investments across many assets so one position, sector, "
                "or country doesn’t dominate results. By combining assets that don’t always move together, "
                "you can reduce the impact of any single loss and smooth portfolio volatility over time. "
                "It doesn’t guarantee profits or prevent losses, but it helps manage risk."
            ),
        ),
        Article(
            title="Market indices explained",
            category="market_fundamentals",
            url="https://example.com/indices",
            content=(
                "A market index is a rules-based basket of securities used to represent a segment of the market "
                "(e.g., large-cap US stocks). Index funds and ETFs aim to track an index by holding its constituents "
                "(or a representative sample). Indices are useful benchmarks for comparing performance and understanding "
                "broad market moves."
            ),
        ),
        Article(
            title="Tax-advantaged accounts overview",
            category="tax_education",
            url="https://example.com/tax-accounts",
            content=(
                "Tax-advantaged accounts (like 401(k)s, IRAs, and HSAs) offer tax benefits to encourage saving. "
                "Some accounts give tax deductions up front (traditional), while others offer tax-free withdrawals "
                "(Roth) if rules are met. Eligibility, contribution limits, and withdrawal rules vary by account type."
            ),
        ),
    ]