import os
import socket
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from summarizer import generate_summary
from discriminator import score_summary

app = FastAPI(title="Text Summarizer API (GAN + NLP)")

# 🔓 CORS Configuration
origins = [
    "http://127.0.0.1:5500",   # Live Server (VS Code)
    "http://localhost:5500",
    "http://127.0.0.1:3000",   # React (if used later)
    "http://localhost:3000",
    "https://text-summarizer-by-alienminus.vercel.app",
    "*"  # ⚠️ Allow all (for development only)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📥 Input schema
class TextInput(BaseModel):
    text: str

# 🏠 Home route
@app.get("/")
def home():
    return {"message": "Summarizer API is running 🚀"}

# 🧠 Summarization route
@app.post("/summarize")
def summarize(input: TextInput):
    text = input.text

    if len(text.strip()) == 0:
        return {"error": "Empty text provided"}

    summary = generate_summary(text)
    score = score_summary(text, summary)

    return {
        "summary": summary,
        "quality_score": score
    }

def get_port(default_port: int = 8000) -> int:
    port = int(os.environ.get("PORT", default_port))
    if port <= 0:
        return default_port
    return port


if __name__ == "__main__":
    port = get_port()
    try:
        uvicorn.run(app, host="0.0.0.0", port=port)
    except OSError as exc:
        if exc.errno == 10048:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                free_port = s.getsockname()[1]
            print(f"Port {port} is busy; retrying on {free_port}")
            uvicorn.run(app, host="127.0.0.1", port=free_port)
        else:
            raise