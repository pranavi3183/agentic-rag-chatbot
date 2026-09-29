from fastapi import FastAPI

from app.graph import rag_graph
from app.schemas import ChatRequest, ChatResponse


app = FastAPI(
    title="Agentic AI RAG Chatbot",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Agentic AI RAG Chatbot is running"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    result = rag_graph.invoke({
        "query": request.query,
        "context": [],
        "answer": "",
        "confidence_score": 0.0
    })

    return {
        "query": request.query,
        "final_answer": result["answer"],
        "retrieved_context_chunks": result["context"],
        "confidence_score": result["confidence_score"]
    }