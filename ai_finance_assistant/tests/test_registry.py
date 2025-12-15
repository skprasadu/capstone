from pathlib import Path

from ai_finance_assistant.src.agents.registry import build_registry, select_agent
from ai_finance_assistant.src.utils.config_loader import load_config


def test_registry_contains_all_agents():
    registry = build_registry()
    assert len(registry) == 6
    assert {agent.name for agent in registry.values()} >= {
        "Finance Q&A Agent",
        "Portfolio Analysis Agent",
        "Market Analysis Agent",
        "Goal Planning Agent",
        "News Synthesizer Agent",
        "Tax Education Agent",
    }


def test_select_agent_routes_by_keyword():
    portfolio_agent = select_agent("Can you review my portfolio allocation?")
    assert portfolio_agent.name == "Portfolio Analysis Agent"
    fallback_agent = select_agent("What is an index fund?")
    assert fallback_agent.name == "Finance Q&A Agent"


def test_load_config_defaults():
    config = load_config(Path("ai_finance_assistant/config.yaml"))
    assert config["app"]["name"] == "AI Finance Assistant"
    assert config["llm"]["provider"] == "openai"
