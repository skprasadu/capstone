# Call Summarizer Agents

A modular, multi-agent pipeline that converts raw call data into summaries and quality insights. The design aligns with the business goals of rapid summarization, scalable QA monitoring, and consistent compliance checks.

## Architecture Overview
- **Call Intake Agent:** validates inputs and extracts metadata.
- **Transcription Agent:** converts audio to text via Whisper/Deepgram (placeholder) or uses provided transcripts.
- **Summarization Agent:** generates concise summaries and key points (LLM-ready with a deterministic fallback).
- **Quality Scoring Agent:** applies a rubric for professionalism, empathy, resolution, and compliance.
- **Routing Agent:** orchestrates handoff, fallbacks, and merges outputs.
- **Streamlit UI:** simple front-end to run the pipeline interactively.

The pipeline is implemented in `call_summarizer_agents/pipeline.py` and orchestrated through the `RoutingAgent`.

## Quickstart (Docker Compose)
1. Copy the environment template and fill in API keys (optional for local testing):
   ```bash
   cp .env.example .env
   # set OPENAI_API_KEY and WHISPER_API_KEY for real model calls
   ```
2. Build and start the Streamlit app:
   ```bash
   docker compose up --build
   ```
3. Open http://localhost:8501. You can upload a `.wav` file for transcription + summarization or paste a chat transcript and ask for a summary/quality review directly in the UI.

## Local Development
1. Create a virtual environment and install dependencies:
   ```bash
   uv venv
   source .venv/bin/activate
   pip install -e .
   ```
2. Run the sample pipeline against the included transcript:
   ```bash
   python main.py
   ```
3. Launch the Streamlit UI:
   ```bash
   streamlit run call_summarizer_agents/ui/streamlit_app.py
   ```

## Configuration
- Copy `.env.example` to `.env` (or set the variables directly) to provide credentials:
  ```bash
  cp .env.example .env
  # then edit .env to set OPENAI_API_KEY and WHISPER_API_KEY
  ```
- `OPENAI_API_KEY` powers the summarization step when available. `WHISPER_API_KEY` enables Whisper audio transcription.
- `call_summarizer_agents/config/mcp.yaml` documents agent entrypoints and environment variables for OpenAI, Whisper, and Langsmith integrations.
- Extend `TranscriptionAgent._transcribe_audio` and `SummarizationAgent._run_openai` to customize API usage.

## Data
Sample transcript stored at `call_summarizer_agents/data/sample_transcripts/sample_call.txt` can be used for demos and tests.

## Testing
- Run the unit suite with pytest:
  ```bash
  pytest
  ```
- Tests cover validation rules, deterministic agent fallbacks, and the full graph pipeline using the sample transcript to ensure the app works without external API keys.

# AI Finance Assistant (Simple)

One textbox. The assistant routes your question to the right agent.
LangGraph stores state so the left sidebar shows conversations + run history.

## 1) Start (Docker Compose)
From repo root:
```bash
cp .env.template .env
# set:
# OPENAI_API_KEY=...
# ALPHA_VANTAGE_API_KEY=...   (only needed for stock quotes)

docker compose up --build
```

Open:
	•	Finance Assistant: http://localhost:8502
	•	Chroma: http://localhost:8000

## 2) Ingest seed docs into Chroma

```bash
docker compose exec ai-finance-assistant python -m ai_finance_assistant.src.rag.ingest --source seed --reset
```

## 3) What to ask (routing map)

| Example question | Routed agent |
| --- | --- |
| What is an index fund? | Finance Q&A Agent |
| Can you review my portfolio allocation? | Portfolio Analysis Agent |
| What does market volatility mean? | Market Analysis Agent |
| Help me plan a 5-year goal to buy a house | Goal Planning Agent |
| Summarize this headline for me | News Synthesizer Agent |
| Explain the difference between a 401k and IRA | Tax Education Agent |
| What is the stock price of IBM? | Stock Quote Agent (Alpha Vantage) |

Notes:
* RAG answers use Chroma (and fallback docs if Chroma/keys are missing).
* Stock quotes require ALPHA_VANTAGE_API_KEY in .env.

## Optional: LangSmith tracing

If you want traces:
  * set LANGSMITH_API_KEY and LANGCHAIN_TRACING_V2=true in .env

``` bash
---

# 7) (Optional but recommended) Fix the test that hardcodes “6 agents”
Your current test expects `len(registry) == 6`. You now have **7** (adds stock_quote).
