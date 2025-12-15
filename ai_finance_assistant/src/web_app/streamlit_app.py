import streamlit as st

from ai_finance_assistant.src.core.architecture import bootstrap_blueprint
from ai_finance_assistant.src.core.disclaimers import FINANCE_DISCLAIMER
from ai_finance_assistant.src.data.knowledge_base import seed_articles
from ai_finance_assistant.src.utils.config_loader import load_config
from ai_finance_assistant.src.workflow.router import route_query

st.set_page_config(page_title="AI Finance Assistant", layout="wide")

config = load_config()
blueprint = bootstrap_blueprint()

st.sidebar.title("Assistant Controls")
st.sidebar.success(config["app"]["disclaimer"])
st.sidebar.markdown("**Environment:** %s" % config["app"].get("environment", "development"))
st.sidebar.markdown("**LLM Provider:** %s" % config["llm"]["provider"])
st.sidebar.markdown("**Vector Store:** %s" % config["rag"]["vector_store"])

st.title("AI Finance Assistant (Prototype)")
st.caption(
    "Multi-agent financial education workspace. Responses are educational only; not financial advice."
)

st.header("Try a prompt")
with st.form("query-form"):
    user_query = st.text_area(
        "Ask a question or describe a goal",
        placeholder="e.g., How should I think about diversification for a retirement portfolio?",
    )
    submitted = st.form_submit_button("Send to Assistant")

if submitted and user_query.strip():
    result = route_query(user_query)
    st.subheader(f"Agent: {result.agent.name}")
    st.markdown(f"**Why routed here:** {result.reasoning}")
    st.markdown("---")
    st.write(result.content)
else:
    st.info("Enter a prompt to see which agent responds and how retrieval is used.")

st.header("Agent roster")
for capability in blueprint.capabilities:
    st.markdown(f"- **{capability.name}:** {capability.description} (Endpoints: {', '.join(capability.endpoints)})")

st.subheader("Registered agents")
for agent in blueprint.agents.values():
    with st.expander(agent.name, expanded=False):
        st.write(agent.description)
        st.write("**Responsibilities:**")
        st.write("\n".join(f"- {item}" for item in agent.responsibilities))
        st.write("**Output style:**", agent.output_format)
        if agent.safety_notes:
            st.warning("\n".join(agent.safety_notes))

st.header("Seed knowledge base")
articles = seed_articles()
st.table({"Title": [a.title for a in articles], "Category": [a.category for a in articles]})

st.caption(f"⚠️ {FINANCE_DISCLAIMER}")
