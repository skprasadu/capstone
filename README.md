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

## Quickstart
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
- `call_summarizer_agents/config/mcp.yaml` documents agent entrypoints and environment variables for OpenAI and Langsmith integrations.
- Extend `TranscriptionAgent._pseudo_transcribe` and `SummarizationAgent._run_llm` to call your preferred APIs.

## Data
Sample transcript stored at `call_summarizer_agents/data/sample_transcripts/sample_call.txt` can be used for demos and tests.
