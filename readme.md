# RAG Evaluation Pipeline

This repository contains a modular and extensible pipeline for experimenting with **Retrieval-Augmented Generation (RAG)** techniques, focused on educational content and small-scale LLMs.  
The project is designed for testing different components independently — retrieval, generation, and evaluation — and combining them into complete RAG workflows.

---

## ✨ Features

- **Vector Store with Qdrant (embedded mode)**
  - Fast similarity search
  - Persistent, local storage
  - Easy collection and point management

- **Flexible Retriever**
  - SentenceTransformer embeddings
  - Stored and indexed with Qdrant
  - Automatic creation of collections

- **Modular Generator**
  - Supports multiple small LLMs (Gemma, Phi, TinyLlama, etc.)
  - Clean prompt formatting for RAG workflows
  - Extensible enum for different models

- **Evaluation Hooks**
  - Ready for integration with Ragas, LLM-as-a-Judge, or custom metrics

- **Fully Local Execution**
  - No external APIs required
  - Easy experimentation on CPU/GPU

---

## 📂 Project Structure

rag-eval/
├── src/
│ ├── retrieval.py # Vector retrieval with Qdrant
│ ├── generation.py # LLM generation component
│ ├── evaluation_ragas.py # RAGAS evaluation (optional)
│ ├── evaluation_judge.py # LLM-judge evaluation (optional)
│ ├── vector_db_manager.py # Qdrant storage manager
│ ├── pipeline.py # Combined RAG pipeline
│ └── main.py # Example usage
├── qdrant_data/ # Local Qdrant storage (ignored in Git)
├── pyproject.toml
└── README.md

## Run It

```bash
poetry install
poetry run python src/main.py