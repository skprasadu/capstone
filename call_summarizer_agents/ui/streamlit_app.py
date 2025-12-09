"""Streamlit UI for orchestrating the call summarization pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from call_summarizer_agents.pipeline import CallSummarizationPipeline

st.set_page_config(page_title="Call Summarizer", layout="wide")
st.title("📞 Call Summarizer & QA Monitor")

st.sidebar.header("Call Metadata")
conversation_id = st.sidebar.text_input("Conversation ID", value="demo-call-001")
agent_name = st.sidebar.text_input("Agent Name", value="Jamie Agent")
customer_name = st.sidebar.text_input("Customer Name", value="Alex Customer")
channel = st.sidebar.selectbox("Channel", ["voice", "chat"], index=0)

uploaded_audio = st.file_uploader("Upload call recording (optional)", type=["wav", "mp3", "txt"])
transcript_text = st.text_area(
    "Paste transcript (optional)",
    value="Hello Alex, thanks for calling. I understand you need a refund for your order...",
    height=200,
)

run_button = st.button("Generate Summary")

if run_button:
    audio_path = None
    if uploaded_audio:
        temp_path = Path(f"/tmp/{uploaded_audio.name}")
        temp_path.write_bytes(uploaded_audio.getvalue())
        audio_path = temp_path

    payload = {
        "conversation_id": conversation_id,
        "agent_name": agent_name,
        "customer_name": customer_name,
        "channel": channel,
        "audio_path": audio_path,
        "transcript": transcript_text or None,
    }

    pipeline = CallSummarizationPipeline()
    with st.spinner("Running multi-agent pipeline..."):
        result = pipeline.run(payload)

    st.success("Pipeline completed")

    st.subheader("Summary")
    st.write(result["summary"]["summary"])

    col1, col2 = st.columns(2)
    col1.metric("Professionalism", result["quality"]["professionalism"])
    col1.metric("Empathy", result["quality"]["empathy"])
    col2.metric("Resolution", result["quality"]["resolution"])
    col2.metric("Compliance", result["quality"]["compliance"])

    st.subheader("Key Points")
    st.write(result["summary"]["key_points"])

    st.subheader("Risks & Follow-ups")
    st.write(result["summary"]["risks"])
    st.write(result["summary"]["follow_ups"])

    st.subheader("Raw Output")
    st.code(json.dumps(result, indent=2, default=str))
else:
    st.info("Provide transcript text or an audio file to begin.")
