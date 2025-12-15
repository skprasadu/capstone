import pytest

from call_summarizer_agents.utils.validation import (
    CallInput,
    ensure_file,
    normalize_transcript_text,
)


def test_call_input_requires_audio_or_transcript():
    with pytest.raises(ValueError):
        CallInput(conversation_id="abc123")


def test_normalize_transcript_text_strips_blank_lines():
    text = "Hello world.\n\n  Thanks for calling.  \n"
    normalized = normalize_transcript_text(text)
    assert normalized == "Hello world.\nThanks for calling."


def test_ensure_file_raises_for_missing_path(tmp_path):
    missing = tmp_path / "missing.wav"
    with pytest.raises(FileNotFoundError):
        ensure_file(missing)
