from __future__ import annotations

import json
import streamlit as st

from ai_finance_assistant.src.pipeline import FinanceAssistantPipeline

st.set_page_config(page_title="AI Finance Assistant", layout="wide")
st.title("AI Finance Assistant")

pipeline = FinanceAssistantPipeline()

# ---- Sidebar: Conversations so far (same UX as call_summarizer) ----
st.sidebar.header("Conversations")

cards = pipeline.list_conversations()
index = {c["conversation_id"]: c for c in cards}

options = ["(new)"] + list(index.keys())

st.session_state.setdefault("selected_conversation", "(new)")

# Safe jump (avoids StreamlitAPIException)
if "_jump_to_conversation" in st.session_state:
    target = st.session_state.pop("_jump_to_conversation")
    st.session_state["selected_conversation"] = target if target in options else "(new)"

selected = st.sidebar.selectbox("Select", options, key="selected_conversation")

if selected != "(new)":
    c = index[selected]
    st.sidebar.caption(
        f'{c.get("agent_name") or "?"}'
        + (f' | {c.get("symbol")}' if c.get("symbol") else "")
    )
    st.sidebar.subheader("Run history")
    for r in reversed(pipeline.get_runs(selected)[-10:]):
        st.sidebar.write(f'{r["at"][:19]}  | {r.get("agent_id")}')

# ---- Main: two experiences ----
if selected == "(new)":
    st.subheader("New question")

    user_query = st.text_area(
        "Ask anything (one box)",
        placeholder="e.g., What is an index fund? / What is the stock price of IBM?",
        height=140,
    )

    send = st.button("Send")

    if send:
        if not user_query.strip():
            st.error("Enter a question.")
            st.stop()

        with st.spinner("Running..."):
            result = pipeline.run({"query": user_query.strip()})

        new_id = result["metadata"]["conversation_id"]
        st.session_state["_jump_to_conversation"] = new_id
        st.rerun()

    st.info("Enter a prompt. The assistant will route to the right agent automatically.")

else:
    st.subheader(f"Conversation: {selected} (read-only)")

    result = pipeline.get_latest_result(selected)
    if not result:
        st.warning("No stored state found for this conversation.")
        st.stop()

    st.subheader("Request")
    st.text_area(
        "Query (read-only)",
        value=(result.get("request") or {}).get("query") or "",
        height=120,
        disabled=True,
    )

    st.subheader("Routing")
    route = result.get("route") or {}
    st.json(route)

    st.subheader("Answer")
    st.write(result.get("answer") or "")

    retrieval = result.get("retrieval") or {}
    docs = retrieval.get("docs") or []
    if docs:
        st.subheader("Retrieved docs")
        st.table(
            {
                "Title": [d.get("title", "") for d in docs],
                "URL": [d.get("url", "") for d in docs],
            }
        )

    market = result.get("market") or {}
    if market:
        st.subheader("Market data")
        st.json(market)

    st.subheader("Raw output")
    st.code(json.dumps(result, indent=2, default=str))