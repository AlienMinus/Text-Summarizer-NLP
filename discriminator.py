import os

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

try:
    from transformers import BertTokenizer, BertForSequenceClassification
    import torch
except Exception:  # pragma: no cover - defensive for older or broken installs
    BertTokenizer = None
    BertForSequenceClassification = None
    torch = None

_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    if BertTokenizer is None or BertForSequenceClassification is None or torch is None:
        return None, None

    try:
        _tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        _model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    except Exception:
        _tokenizer = None
        _model = None

    return _tokenizer, _model


def score_summary(text: str, summary: str) -> float:
    if not text or not summary:
        return 0.5

    tokenizer, model = _load_model()
    if tokenizer is None or model is None or torch is None:
        return 0.5

    try:
        combined = text + " [SEP] " + summary

        inputs = tokenizer(
            combined,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)
        return round(probs[0][1].item(), 4)

    except Exception:
        return 0.5