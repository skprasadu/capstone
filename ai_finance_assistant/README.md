# AI Finance Assistant

The AI Finance Assistant is a new multi-agent workspace focused on financial education, portfolio analysis, and market context. This directory contains the initial scaffolding, configuration, and UI entrypoint for developing the production-ready experience described in the project requirements.

## Highlights
- Six specialized agents (Finance Q&A, Portfolio Analysis, Market Analysis, Goal Planning, News Synthesizer, and Tax Education) registered in a shared registry for orchestration.
- Streamlit-based prototype UI (`ai_finance_assistant/src/web_app/streamlit_app.py`) that surfaces agent capabilities and provides an interactive playground.
- YAML-based configuration for API keys, feature toggles, and RAG settings.
- Shared Python dependencies with the root `pyproject.toml` to maximize reuse.

## Project Layout
```
ai_finance_assistant/
├── config.yaml
├── requirements.txt
├── README.md
├── src/
│   ├── agents/
│   ├── core/
│   ├── data/
│   ├── rag/
│   ├── utils/
│   ├── web_app/
│   └── workflow/
└── tests/
```

## Running with Docker Compose
Two services are defined in `docker-compose.yml`:
- `call-summarizer` (existing app) uses `Dockerfile.cca`.
- `ai-finance-assistant` (this project) uses `Dockerfile.fa` and exposes port `8502`.

Launch both with:
```bash
docker-compose up --build
```
Then open http://localhost:8502 for the finance assistant UI.

## Next Steps
- Connect real LLM providers and vector databases via the stubs in `rag/pipeline.py` and `workflow/router.py`.
- Expand the knowledge base under `data/` with curated articles and add embedding/indexing logic.
- Add rigorous test coverage (target 80%+) as more functionality is implemented.
- Ensure regulatory disclaimers remain visible and accurate throughout the user experience.
