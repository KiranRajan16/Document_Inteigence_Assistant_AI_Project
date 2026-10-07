# AI Research and Document Assistant

An AI-powered document research and question-answering application built using Python, LangChain, LangGraph, RAG, ChromaDB, Groq LLM, FastAPI, LangServe, LangSmith, and Streamlit.

The application allows users to upload PDF documents, ask questions, retrieve relevant information, generate grounded answers, summarize documents, and compare documents.

---

## Project Overview

The AI Research and Document Assistant combines Retrieval-Augmented Generation (RAG) with LangGraph-based workflow orchestration to provide document-grounded responses.

The system processes uploaded PDF documents, converts them into text chunks, generates embeddings, stores them in a Chroma vector database, retrieves relevant context for a user query, and uses an LLM to generate an answer.

A grounding-check mechanism is included to reduce unsupported or hallucinated responses.

---

## Objectives

- Upload and process PDF documents
- Extract and split document text into manageable chunks
- Generate vector embeddings
- Store document embeddings in ChromaDB
- Retrieve relevant document context
- Generate answers using an LLM
- Reduce hallucinations using grounding verification
- Support document summarization
- Support document comparison
- Implement a state-based LangGraph workflow
- Monitor LLM execution using LangSmith
- Expose the RAG workflow through FastAPI and LangServe
- Provide an interactive Streamlit interface

---

## System Architecture

```text
                    ┌──────────────────────┐
                    │        User          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Streamlit       │
                    │    User Interface    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Document Processing│
                    │   PDF → Text → Chunks│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Embeddings      │
                    │ all-MiniLM-L6-v2     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      ChromaDB        │
                    │    Vector Database   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Retriever       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      LangGraph       │
                    │ Retrieve → Generate  │
                    │ → Grounding Check    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Groq LLM        │
                    │   GPT-OSS-120B       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Answer + Sources     │
                    └──────────────────────┘

       LangSmith → Tracing, Monitoring & Evaluation
       FastAPI   → REST API
       LangServe → RAG API / Playground