from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model once
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

def score_summary(text: str, summary: str) -> float:
    try:
        combined = text + " [SEP] " + summary

        inputs = tokenizer(
            combined,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)

        # Return probability of "realistic"
        return round(probs[0][1].item(), 4)

    except Exception:
        return 0.5  # fallback neutral score