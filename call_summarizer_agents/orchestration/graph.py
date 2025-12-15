from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from call_summarizer_agents.agents.intake_agent import CallIntakeAgent
from call_summarizer_agents.agents.transcription_agent import TranscriptionAgent
from call_summarizer_agents.agents.summarization_agent import SummarizationAgent
from call_summarizer_agents.agents.quality_score_agent import QualityScoreAgent
from call_summarizer_agents.config.settings import AppSettings, load_settings
from call_summarizer_agents.orchestration.state import CallState, RunRecord
from call_summarizer_agents.utils.debug import dlog


class CallSummarizerGraph:
    """
    LangGraph-based orchestrator.
    - State management: CallState in the graph
    - Memory: checkpointer + thread_id (conversation_id) and a reducer-based `runs` list
    """

    def __init__(
        self,
        settings: AppSettings | None = None,
        intake_agent: Optional[CallIntakeAgent] = None,
        transcription_agent: Optional[TranscriptionAgent] = None,
        summarization_agent: Optional[SummarizationAgent] = None,
        quality_agent: Optional[QualityScoreAgent] = None,
        checkpointer: Any | None = None,
    ) -> None:
        self.settings = settings or load_settings()

        self.intake_agent = intake_agent or CallIntakeAgent()

        whisper_key = self.settings.whisper_api_key or self.settings.openai_api_key
        self.transcription_agent = transcription_agent or TranscriptionAgent(
            whisper_api_key=whisper_key,
            whisper_model=self.settings.whisper_model,
        )

        self.summarization_agent = summarization_agent or SummarizationAgent(
            openai_api_key=self.settings.openai_api_key,
            openai_model=self.settings.openai_model,
            temperature=self.settings.openai_temperature,
        )

        self.quality_agent = quality_agent or QualityScoreAgent(
            openai_api_key=self.settings.openai_api_key,
            openai_model=self.settings.openai_model,
            temperature=0.0,
        )

        # Memory/persistence layer (in-process). For durable memory, swap to sqlite/postgres later.
        self.checkpointer = checkpointer or InMemorySaver()

        workflow = self._build()
        self.graph = workflow.compile(checkpointer=self.checkpointer)

    def _build(self) -> StateGraph:
        g = StateGraph(CallState)

        g.add_node("intake", self._node_intake)
        g.add_node("transcribe", self._node_transcribe)
        g.add_node("summarize", self._node_summarize)
        g.add_node("quality", self._node_quality)
        g.add_node("finalize", self._node_finalize)

        g.add_edge(START, "intake")
        g.add_edge("intake", "transcribe")
        g.add_edge("transcribe", "summarize")
        g.add_edge("summarize", "quality")
        g.add_edge("quality", "finalize")
        g.add_edge("finalize", END)

        return g

    # -------------------------
    # Nodes
    # -------------------------

    def _node_intake(self, state: CallState) -> dict[str, Any]:
        raw = state.get("raw_payload") or {}
        dlog("langgraph.intake.start", keys=list(raw.keys()))

        intake_model = self.intake_agent(raw)
        metadata = self.intake_agent.extract_metadata(intake_model)

        intake_dict = intake_model.model_dump(mode="json")

        dlog(
            "langgraph.intake.ok",
            has_audio=metadata.get("has_audio"),
            has_transcript=metadata.get("has_transcript"),
        )

        return {
            "intake": intake_dict,
            "metadata": metadata,
        }

    def _node_transcribe(self, state: CallState) -> dict[str, Any]:
        intake = state["intake"]
        dlog("langgraph.transcribe.start", conversation_id=intake.get("conversation_id"))

        transcript_model = self.transcription_agent(intake)
        transcript_dict = transcript_model.model_dump(mode="json")

        dlog("langgraph.transcribe.ok", text_len=len(transcript_dict.get("transcript") or ""))

        return {"transcript": transcript_dict}

    def _node_summarize(self, state: CallState) -> dict[str, Any]:
        transcript = state["transcript"]
        dlog("langgraph.summarize.start", conversation_id=transcript.get("conversation_id"))

        summary_model = self.summarization_agent(transcript)
        summary_dict = summary_model.model_dump(mode="json")

        dlog("langgraph.summarize.ok", summary_len=len(summary_dict.get("summary") or ""))

        return {"summary": summary_dict}

    def _node_quality(self, state: CallState) -> dict[str, Any]:
        merged = {**state["transcript"], **state["summary"]}
        dlog("langgraph.quality.start", conversation_id=merged.get("conversation_id"))

        quality_model = self.quality_agent(merged)
        quality_dict = quality_model.model_dump(mode="json")

        dlog("langgraph.quality.ok", overall=quality_dict.get("overall"))

        return {"quality": quality_dict}

    def _node_finalize(self, state: CallState) -> dict[str, Any]:
        result = {
            "metadata": state["metadata"],
            "transcript": state["transcript"],
            "summary": state["summary"],
            "quality": state["quality"],
        }

        run_record: RunRecord = {
            "at": datetime.now(timezone.utc).isoformat(),
            "conversation_id": state["metadata"].get("conversation_id") or "unknown",
            "summary": state["summary"].get("summary") or "",
            "overall": state["quality"].get("overall"),
        }

        # IMPORTANT:
        # returning runs as a list (single item) works with reducer `add` to append over time.
        return {
            "result": result,
            "runs": [run_record],
        }

    # -------------------------
    # Public API
    # -------------------------

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        conversation_id = str(payload.get("conversation_id") or "unknown").strip()
        config = {
            "configurable": {"thread_id": conversation_id},
            "tags": ["call-summarizer", str(payload.get("channel") or "unknown")],
            "metadata": {
                "conversation_id": conversation_id,
                "has_audio": bool(payload.get("audio_path")),
                "has_transcript": bool(payload.get("transcript")),
            },
            "run_name": "call-summarizer-graph",
        }

        # thread_id is what the checkpointer uses to store/retrieve state and history  [oai_citation:3‡LangChain Docs](https://docs.langchain.com/oss/python/langgraph/persistence)
        final_state: CallState = self.graph.invoke({"raw_payload": payload}, config)
        return final_state["result"]

    def get_runs(self, conversation_id: str) -> list[RunRecord]:
        config = {"configurable": {"thread_id": conversation_id}}
        snapshot = self.graph.get_state(config)  # latest checkpoint  [oai_citation:4‡LangChain Docs](https://docs.langchain.com/oss/python/langgraph/persistence)
        return list(snapshot.values.get("runs") or [])

    def get_state_history(self, conversation_id: str) -> list[Any]:
        config = {"configurable": {"thread_id": conversation_id}}
        return list(self.graph.get_state_history(config))  # full checkpoint history  [oai_citation:5‡LangChain Docs](https://docs.langchain.com/oss/python/langgraph/persistence)