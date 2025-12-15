from dataclasses import dataclass
from typing import List


@dataclass
class Article:
    title: str
    category: str
    url: str


def seed_articles() -> List[Article]:
    return [
        Article(
            title="Why diversification matters",
            category="portfolio_basics",
            url="https://example.com/diversification",
        ),
        Article(
            title="Market indices explained",
            category="market_fundamentals",
            url="https://example.com/indices",
        ),
        Article(
            title="Tax-advantaged accounts overview",
            category="tax_education",
            url="https://example.com/tax-accounts",
        ),
    ]
