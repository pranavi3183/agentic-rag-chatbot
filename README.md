# Agentic AI RAG Chatbot

A production-style Retrieval-Augmented Generation (RAG) chatbot built using LangGraph, Pinecone, Gemini Embeddings, NVIDIA NIM, and FastAPI.

## Overview

This project answers questions strictly using the content available in the provided **Agentic AI eBook**.

The system follows a RAG workflow:

1. Load the Agentic AI PDF
2. Split the document into chunks
3. Generate embeddings using Gemini
4. Store embeddings in Pinecone
5. Retrieve relevant context for a user query
6. Generate an answer using NVIDIA NIM
7. Calculate a confidence score
8. Return a structured JSON response through FastAPI

## Architecture

```text
User Query
    ↓
FastAPI
    ↓
LangGraph
    ↓
Retrieve Node
    ↓
Pinecone Vector Store
    ↓
Relevant Context
    ↓
Generate Node
    ↓
NVIDIA NIM
    ↓
Confidence Node
    ↓
Structured JSON Response