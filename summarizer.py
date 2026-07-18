import os
import re

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - defensive for older or broken installs
    pipeline = None

_summarizer = None


def _load_summarizer():
    global _summarizer
    if _summarizer is not None:
        return _summarizer

    if pipeline is None:
        return None

    try:
        _summarizer = pipeline(
            task="summarization",
            model="facebook/bart-large-cnn",
            device=-1,
        )
    except Exception:
        _summarizer = None

    return _summarizer


def _fallback_summary(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    if not sentences:
        return cleaned

    if len(sentences) <= 2:
        return sentences[0]

    return " ".join(sentences[:2]).strip()


def generate_summary(text: str) -> str:
    if not text or not text.strip():
        return ""

    try:
        summarizer = _load_summarizer()
        if summarizer is None:
            raise RuntimeError("Summarization model unavailable")

        result = summarizer(
            text,
            max_length=130,
            min_length=30,
            do_sample=False,
        )
        if result and isinstance(result, list) and result[0].get("summary_text"):
            return result[0]["summary_text"]
    except Exception:
        pass

    return _fallback_summary(text)