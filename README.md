# Advanced Agentic RAG System

This project is an advanced **Retrieval-Augmented Generation (RAG)** system built using **LlamaIndex** and **Ollama**. It implements an agentic workflow that plans, retrieves, and synthesizes information from multiple PDF books to answer complex user queries.

## 🚀 Key Features

- **Agentic Workflow**: distinct layers for Planning, Decomposition, Retrieval, and Synthesis.
- **Multi-Book Support**: Intelligently selects relevant books based on query intent using a `QueryPlanner`.
- **Hybrid Search**: Combines **Vector Search** (Dense) and **BM25** (Sparse) using `QueryFusionRetriever` for robust retrieval.
- **Reranking**: integrated `SentenceTransformerRerank` to improve context quality.
- **Explicit Retrieval & Grounding**: Separates retrieval from reasoning to ensure answers are strictly grounded in retrieved context (minimizing hallucinations).
- **Evaluation**: Integrated **RAGAS** pipeline for evaluating Faithfulness, Answer Relevancy, Context Precision, and Answer Correctness.
- **Memory**: `AgenticMemory` tracks successful queries and book usage patterns.

## 🏗️ Architecture

The system is orchestrated by `AgentOrchestrator` in `advanced_rag_agentic.py`:

1.  **Planning**: `QueryPlanner` analyzes the text to determine which books are needed and if the query needs decomposition.
2.  **Decomposition**: `QueryDecomposer` breaks down complex questions into sub-queries if necessary.
3.  **Execution (Step-by-Step)**:
    - **Retrieval**: Explicitly queries the selected `QueryEngineTool` (Hybrid Search) for relevant context.
    - **Synthesis**: Uses the LLM (directly or via Agent) to generate an answer _only_ using the retrieved context.
4.  **Validation**: `AnswerValidator` checks if the answer adequately addresses the question.
5.  **Refinement**: If validation fails, it retries with feedback.

## 🛠️ Components

- **`advanced_rag_agentic.py`**: The main application entry point. Contains the agentic logic (`Librarian`, `AgentOrchestrator`, etc.).
- **`utils.py`**: Utilities for:
  - **PDF Parsing**: Smart chunking of PDFs by chapters using `PyMuPDF` (`fitz`).
  - **Evaluation**: Wrappers for running RAGAS metrics.
- **`configs/`**:
  - `books_config.yaml`: Register your PDF books (paths, descriptions, keywords).
  - `models_config.yaml`: Configure LLM (Ollama) and Embedding models.

## ⚙️ Installation

1.  **Prerequisites**:

    - Python 3.10+
    - [Ollama](https://ollama.com/) installed and running.

2.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Models**:
    - Pull the LLM model in Ollama (default: `qwen2.5:7b-instruct` or as configured in `models_config.yaml`):
      ```bash
      ollama pull qwen2.5:7b-instruct
      ```

## 📖 Configuration

1.  **Add Books**:
    Edit `configs/books_config.yaml` to include your PDF files.

    ```yaml
    books:
      - name: "MyBook"
        pdf_path: "./files/MyBook.pdf"
        description: "A description of the book content."
        keywords: ["keyword1", "keyword2"]
        enabled: true
    ```

2.  **Model Config**:
    Adjust model parameters in `configs/models_config.yaml` if needed (e.g., context window, temperature).

## 🏃 Usage

Run the agentic RAG system:

```bash
python advanced_rag_agentic.py
```

This will:

1.  Initialize the `Librarian`.
2.  Build/Load indexes for enabled books.
3.  Read evaluation questions from `./files/BuildTrap/eval_questions.json` (default path in `main`).
4.  Generate answers.
5.  Run RAGAS evaluation.
6.  Save results to `./files/BuildTrap/generated_answer/agentic_rag.json`.

## 🧠 Memory & caching

- Indexes are cached in `./cache/indexes` to speed up subsequent runs.
- Agent memory is stored in `./cache/memory/agent_memory.json`.
