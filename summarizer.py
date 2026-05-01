from transformers import pipeline

# Load once (IMPORTANT for performance)
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=-1  # change to 0 if GPU available
)

def generate_summary(text: str) -> str:
    try:
        result = summarizer(
            text,
            max_length=130,
            min_length=30,
            do_sample=False
        )
        return result[0]['summary_text']
    except Exception as e:
        return f"Error generating summary: {str(e)}"