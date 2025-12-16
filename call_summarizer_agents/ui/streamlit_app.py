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

options = ["(new)"] + list(index.keys())

st.session_state.setdefault("selected_conversation", "(new)")

# Safe "jump" pattern to avoid StreamlitAPIException
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

    # ChatGPT-like composer row: [+] + one text area + Send
    col_attach, col_text, col_send = st.columns([1, 10, 1], vertical_alignment="bottom")

    with col_attach:
        # Popover keeps the UI clean (no giant dropzone on screen)
        with st.popover("➕", help="Attach audio (.wav/.mp3)"):
            uploaded_audio = st.file_uploader(
                "Attach audio",
                type=["wav", "mp3"],
                accept_multiple_files=False,
                key="composer_attachment",
                label_visibility="collapsed",
            )
            if uploaded_audio:
                st.caption(f"Attached: {uploaded_audio.name}")

    with col_text:
        transcript_text = st.text_area(
            "Message",
            value="",
            height=140,
            placeholder="Paste transcript here… or click ➕ to attach audio.",
            key="composer_text",
            label_visibility="collapsed",
        )

    with col_send:
        run_button = st.button("Send", use_container_width=True)

    if run_button:
        typed = (transcript_text or "").strip()
        attached = st.session_state.get("composer_attachment")

        if not typed and not attached:
            st.error("Paste transcript text or attach an audio file.")
            st.stop()

        audio_path = None
        if attached:
            data = attached.getvalue()
            dlog("ui.upload", filename=attached.name, uploaded_bytes=len(data))
            temp_path = Path(f"/tmp/{attached.name}")
            temp_path.write_bytes(data)
            audio_path = temp_path

        # Rule: transcript wins (if you paste text + attach audio, we use the pasted text)
        payload = {
            "channel": "voice" if (audio_path and not typed) else "chat",
            "audio_path": audio_path if not typed else None,
            "transcript": typed or None,
        }

        with st.spinner("Running pipeline..."):
            result = pipeline.run(payload)

        new_id = result["metadata"]["conversation_id"]
        st.session_state["_jump_to_conversation"] = new_id
        st.rerun()

else:
    st.subheader(f"Conversation: {selected} (read-only)")

    result = pipeline.get_latest_result(selected)
    if not result:
        st.warning("No stored state found for this conversation.")
        st.stop()

    st.subheader("Metadata")
    st.json(result.get("metadata", {}))

    st.subheader("Transcript")
    transcript = (result.get("transcript") or {}).get("transcript") or ""
    st.write(transcript)

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