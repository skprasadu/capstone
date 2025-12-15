from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from call_summarizer_agents.pipeline import CallSummarizationPipeline
from call_summarizer_agents.utils.debug import dlog

st.set_page_config(page_title="Call Summarizer", layout="wide")
st.title("Call Summarizer & QA Monitor")

pipeline = CallSummarizationPipeline()

# ---- Sidebar: Conversations so far ----
st.sidebar.header("Conversations")

cards = pipeline.list_conversations()
index = {c["conversation_id"]: c for c in cards}

cards = pipeline.list_conversations()
index = {c["conversation_id"]: c for c in cards}

options = ["(new)"] + list(index.keys())

st.session_state.setdefault("selected_conversation", "(new)")

if "_jump_to_conversation" in st.session_state:
    target = st.session_state.pop("_jump_to_conversation")
    st.session_state["selected_conversation"] = target if target in options else "(new)"

selected = st.sidebar.selectbox("Select", options, key="selected_conversation")

if selected != "(new)":
    c = index[selected]
    st.sidebar.caption(
        f'{c.get("agent_name") or "?"}  {c.get("customer_name") or "?"} | '
        f'{c.get("channel")} | overall={c.get("overall")}'
    )
    st.sidebar.subheader("Run history")
    for r in reversed(pipeline.get_runs(selected)[-10:]):
        st.sidebar.write(f'{r["at"][:19]}  | overall={r.get("overall")}')

# ---- Main: two experiences ----
if selected == "(new)":
    st.subheader("New run")

    uploaded_audio = st.file_uploader("Upload recording (optional)", type=["wav", "mp3", "txt"])
    transcript_text = st.text_area("Paste transcript (optional)", value="", height=220)

    run_button = st.button("Generate Summary")

    if run_button:
        if not uploaded_audio and not transcript_text.strip():
            st.error("Provide transcript text or upload an audio/text file.")
            st.stop()

        audio_path = None
        if uploaded_audio:
            data = uploaded_audio.getvalue()
            dlog("ui.upload", filename=uploaded_audio.name, uploaded_bytes=len(data))
            temp_path = Path(f"/tmp/{uploaded_audio.name}")
            temp_path.write_bytes(data)
            audio_path = temp_path

        payload = {
            "channel": "voice" if audio_path else "chat",
            "audio_path": audio_path,
            "transcript": transcript_text.strip() or None,
        }

        with st.spinner("Running pipeline..."):
            result = pipeline.run(payload)

        # Switch UX to read-only view of the created conversation
        new_id = result["metadata"]["conversation_id"]
        st.session_state["_jump_to_conversation"] = new_id
        st.rerun()

    st.info("Start a new run by pasting a transcript or uploading audio.")

else:
    st.subheader(f"Conversation: {selected} (read-only)")

    result = pipeline.get_latest_result(selected)
    if not result:
        st.warning("No stored state found for this conversation.")
        st.stop()

    # Show everything: metadata + transcript + summary + quality + raw json
    st.subheader("Metadata")
    st.json(result.get("metadata", {}))

    st.subheader("Transcript")
    transcript = (result.get("transcript") or {}).get("transcript") or ""
    st.text_area("Transcript (read-only)", value=transcript, height=320, disabled=True)

    st.subheader("Summary")
    st.write((result.get("summary") or {}).get("summary") or "")

    q = result.get("quality") or {}
    col1, col2 = st.columns(2)
    col1.metric("Professionalism", q.get("professionalism", 0))
    col1.metric("Empathy", q.get("empathy", 0))
    col2.metric("Resolution", q.get("resolution", 0))
    col2.metric("Compliance", q.get("compliance", 0))

    st.subheader("Raw Output")
    st.code(json.dumps(result, indent=2, default=str))