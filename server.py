"""Local Qwen API.

Run: python -m uvicorn server:app --host 127.0.0.1 --port 8000 --workers 1
"""

from contextlib import asynccontextmanager
from threading import Lock
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

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


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)

    @field_validator("messages")
    @classmethod
    def validate_conversation(cls, messages: list[ChatMessage]) -> list[ChatMessage]:
        for index, message in enumerate(messages):
            expected_role = "user" if index % 2 == 0 else "assistant"
            if message.role != expected_role:
                raise ValueError("Messages must alternate user and assistant, starting with user")
            if not message.content.strip():
                raise ValueError("Message content must not be blank")
        if messages[-1].role != "user":
            raise ValueError("The last message must be from the user")
        return messages


@app.post("/chat")
def chat(payload: ChatRequest, request: Request) -> dict[str, str]:
    # History belongs to this request; the server does not store conversations.
    messages = [message.model_dump() for message in payload.messages]

    # Allow only one generation at a time on the shared model.
    with request.app.state.generation_lock:
        response = generate_response(
            request.app.state.tokenizer,
            request.app.state.model,
            messages,
        )

    return {"response": response}
