from pydantic import BaseModel
from typing import List


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: List[str]
    confidence_score: float