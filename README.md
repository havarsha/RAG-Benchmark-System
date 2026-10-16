# RAG Benchmark System

An AI-powered benchmarking system for evaluating and comparing Retrieval-Augmented Generation (RAG) pipelines using different embedding models, retrieval strategies, and Large Language Models (LLMs).

## Overview

The **RAG Benchmark System** is designed to analyze how effectively a RAG pipeline retrieves relevant information and generates answers based on a given knowledge source.

It explores different retrieval approaches, embedding models, and LLMs to understand their impact on answer quality, relevance, and performance. The system also incorporates hybrid retrieval and reranking techniques to improve the retrieval process.

The project provides an interactive way to experiment with RAG configurations and compare their evaluation results.

## Key Features

- **RAG Pipeline Evaluation:** Evaluate retrieval and answer-generation performance.
- **Embedding Model Comparison:** Experiment with different embedding models, including MiniLM and BGE.
- **LLM Comparison:** Test supported language models such as Llama and Gemma.
- **Hybrid Retrieval:** Combine BM25 keyword-based retrieval with vector similarity search.
- **Reranking:** Reorder retrieved documents to improve the relevance of the final context.
- **Memory-Based Retrieval:** Experiment with memory-enabled retrieval for maintaining relevant context.
- **Performance Metrics:** Compare configurations using available retrieval and answer-evaluation metrics.
- **Interactive Interface:** Explore RAG configurations and evaluation results through a Streamlit interface.

## Technologies Used

- **Python** – Core programming language
- **Streamlit** – Interactive web interface
- **Retrieval-Augmented Generation (RAG)** – Knowledge-grounded response generation
- **ChromaDB** – Vector database and similarity search
- **Sentence Transformers** – Text embeddings
- **BM25** – Keyword-based information retrieval
- **Large Language Models (LLMs)** – Response generation
- **Groq API** – LLM inference for supported configurations

## System Architecture

The system follows a typical RAG workflow:

1. **Knowledge Input:** Provide the knowledge documents or data used by the pipeline.
2. **Document Processing:** Prepare the content for retrieval.
3. **Embedding Generation:** Convert text into vector representations.
4. **Indexing:** Store document embeddings in a vector database.
5. **Retrieval:** Retrieve relevant documents using vector search, BM25, or hybrid retrieval.
6. **Reranking:** Improve the ordering of retrieved documents when enabled.
7. **Answer Generation:** Pass the relevant context and user query to a supported LLM.
8. **Evaluation:** Analyze and compare the results using the metrics implemented in the system.

## Models and Retrieval Approaches

### Embedding Models
- MiniLM
- BGE

### Language Models
- Llama-family models
- Gemma-family models

*Exact model availability depends on the configurations supported by the implementation.*

### Retrieval Techniques
- Vector similarity search
- BM25 keyword retrieval
- Hybrid retrieval
- Reranking
- Memory-based retrieval

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version supported by the dependencies
- pip
- A Groq API key if using Groq-hosted models

### 1. Clone the Repository

```bash
git clone https://github.com/havarsha/RAG-Benchmark-System.git
cd RAG-Benchmark-System
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

If the repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Otherwise, install the dependencies listed in the project's setup instructions.

### 4. Configure Environment Variables

Create a `.env` file in the project root if required by the implementation.

```env
GROQ_API_KEY=your_groq_api_key
```

Replace the placeholder with your own API key. Never commit your `.env` file or expose API keys in your repository.

### 5. Run the Application

If the Streamlit entry-point file is `app.py`, run:

```bash
streamlit run app.py
```

If your main file has a different name, replace `app.py` with the correct filename.

## Evaluation

The system is intended to compare RAG configurations based on the evaluation metrics implemented in the project. Depending on the selected configuration, these may include:

- Retrieval relevance
- Context quality
- Answer relevance
- Response quality
- Comparative performance across different pipelines

The actual metrics and their interpretation depend on the evaluation methods configured in the implementation.

## Use Cases

- Comparing different RAG pipeline configurations
- Experimenting with embedding models and LLMs
- Studying hybrid retrieval and reranking techniques
- Evaluating knowledge-grounded AI responses
- Understanding the impact of retrieval choices on generated answers

## Future Enhancements

- Add support for additional embedding models and LLMs.
- Expand automated evaluation using standard RAG evaluation frameworks.
- Add detailed latency and resource-usage measurements.
- Introduce benchmark datasets and reproducible evaluation runs.
- Improve result visualization and exportable benchmark reports.

## Learning Outcomes

Through this project, I explored:

- Retrieval-Augmented Generation architecture
- Vector databases and semantic search
- Embedding models and document retrieval
- Hybrid search and reranking
- LLM integration
- Evaluation and comparison of AI pipelines

## Author

**Varsha H A**

- GitHub: https://github.com/havarsha
- Project Repository: https://github.com/havarsha/RAG-Benchmark-System

---

*This project is developed for learning, experimentation, and evaluation of Retrieval-Augmented Generation systems.*
