from typing import TypedDict, List

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

from app.config import NVIDIA_API_KEY
from app.vector_store import create_vector_store


class RAGState(TypedDict):
    query: str
    context: List[str]
    answer: str
    confidence_score: float


vector_store = create_vector_store()

retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)


llm = ChatOpenAI(
    model="z-ai/glm-5.3-flash",
    temperature=0,
    api_key=NVIDIA_API_KEY,
    base_url="https://integrate.api.nvidia.com/v1",
    timeout=30,
    max_retries=0,
    max_tokens=150,
    reasoning_effort="low",
)


def retrieve_node(state: RAGState):
    documents = retriever.invoke(state["query"])

    context = [
        document.page_content
        for document in documents
    ]

    return {
        "context": context
    }


def generate_node(state: RAGState):
    context = "\n\n".join(state["context"])

    prompt = f"""
Answer ONLY from this context.

Context:
{context}

Question:
{state["query"]}
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


def confidence_node(state: RAGState):
    answer = str(state["answer"]).lower()

    if (
        "does not contain information" in answer
        or "cannot answer that question" in answer
        or "don't have enough information" in answer
        or "not available in the context" in answer
    ):
        score = 0.10
    else:
        score = 0.90

    return {
        "confidence_score": score
    }


workflow = StateGraph(RAGState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("confidence", confidence_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "confidence")
workflow.add_edge("confidence", END)

rag_graph = workflow.compile()