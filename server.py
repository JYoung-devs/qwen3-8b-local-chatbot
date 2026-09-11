"""Local Qwen API.

Run: python -m uvicorn server:app --host 127.0.0.1 --port 8000 --workers 1
"""

from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from run_qwen import generate_response, load_chatbot


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup and reuse the same model for every request.
    app.state.tokenizer, app.state.model = load_chatbot()
    app.state.generation_lock = Lock()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    # Only the standalone frontend served locally on port 5500 may connect.
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(payload: ChatRequest, request: Request) -> dict[str, str]:
    # Each request starts a fresh conversation in Qwen's existing message format.
    messages = [{"role": "user", "content": payload.message}]

    # Allow only one generation at a time on the shared model.
    with request.app.state.generation_lock:
        response = generate_response(
            request.app.state.tokenizer,
            request.app.state.model,
            messages,
        )

    return {"response": response}
