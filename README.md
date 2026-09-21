# Document Q&A — RAG System

A document question-answering system built using Retrieval-Augmented Generation (RAG).

The application retrieves relevant information from a local document collection and uses a local LLM to generate answers grounded in the retrieved context.

## Features

- Supports TXT, Markdown, and PDF documents
- Semantic search using Sentence Transformers
- Local vector database using ChromaDB
- Local LLM inference using Ollama
- LangGraph-based RAG workflow
- Question routing for document and general questions
- Retrieval relevance checking
- Answer verification and retry logic
- Source document display
- Streamlit web interface
- Retrieval and end-to-end evaluation

## Architecture

```text
Documents
    |
    v
Document Loader
    |
    v
Text Chunking
    |
    v
Sentence Transformers
    |
    v
ChromaDB Vector Store
    |
    v
Semantic Retrieval
    |
    v
Retrieval Relevance Check
    |
    +---- Not Relevant ---> "I don't know..."
    |
    v
Answer Generation
    |
    v
Answer Verification
    |
    +---- Not Supported ---> Retry
    |
    v
Answer + Sources
