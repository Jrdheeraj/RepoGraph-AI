# RepoGraph AI

RepoGraph AI is an advanced, AI-driven codebase analysis, indexing, and Graph-RAG (Retrieval-Augmented Generation) platform. By scanning and parsing source repositories, it constructs detailed code syntax graphs, extracts vital structural metrics, and utilizes a multi-agent LangGraph workflow alongside ML inference models to predict code quality, maintainability, technical debt, and support semantic code searches.

---

## 🚀 Key Features

*   **Repository Scanner & AST Parser**: Recursively crawls Python projects, parses files using `tree-sitter` and `LibCST`, and extracts semantic metadata for functions, classes, methods, and decorators.
*   **Semantic Chunking**: Splits codebases into logical, symbol-based chunks (classes, modules, functions) rather than arbitrary text lines.
*   **Dependency Graph Builder**: Analyzes imports and references to construct a directional graph mapping relationships across files.
*   **Agentic Hybrid RAG**: Uses a dual-retrieval engine (Dense FAISS embeddings + Sparse BM25 keyword matching) along with a reranking step. A LangGraph agentic router coordinates specialized agents (Architect, Documentation, Test, Refactoring) to generate contextual answers.
*   **Machine Learning Predictors**: Integrates models to calculate code health metrics (`iq_score`, `maintainability_risk`, `technical_debt_score`, `architecture_quality`) and compute an aggregated `repograph_score`.
*   **FastAPI REST API**: Scalable backend exposing a code analysis pipeline directly via HTTP endpoints.

---

## 📁 Repository Structure

```text
├── agents/            # LangGraph multi-agent workflows and specialized nodes (Architect, Documentation, etc.)
├── ai_models/         # ML model configurations and artifacts
├── backend/           # FastAPI application (endpoints, services, database migrations, configuration)
├── config/            # System-wide configuration templates
├── data/              # Workspace for database storage, cloned repositories, and cache files
├── docker/            # Docker containers and docker-compose orchestration
├── docs/              # Architectural design plans and developer documentation
├── examples/          # Integration examples and guides
├── frontend/          # (Placeholder) React/Next.js/Vite user interface
├── logs/              # System execution logs
├── memory/            # Persistence managers for agents' session history
├── ml/                # ML predictor services, feature extractors, and model training scripts
├── parser/            # Core code parsers, repository scanners, and AST chunkers
├── rag/               # Retrieval pipeline (embedding generation, dense/sparse retrieval, rerankers)
├── requirements.txt   # Project dependencies
├── scripts/           # Utility, database, and setup scripts
└── tests/             # Comprehensive pytest suite
```

---

## 🛠️ Getting Started

### Prerequisites

*   Python 3.10+
*   PostgreSQL database (optional; defaults to a cloud instance in development)
*   GitHub Personal Access Token (optional; for private repositories)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd RepoGraph-AI
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables**:
    Create a `.env` file inside the `backend` directory (or workspace root as required) and configure:
    ```env
    DATABASE_URL=postgresql://user:password@host/dbname
    SECRET_KEY=your_secret_key
    ENVIRONMENT=development
    GITHUB_TOKEN=your_github_token
    ```

### Running the Application

To run the FastAPI backend server local development instance:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Once running, access the interactive API docs at:
*   Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
*   ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🔌 API Usage

### Analyze a Repository

Performs structural parsing, metric extraction, and ML scoring on a given repository.

*   **URL**: `/api/analyze`
*   **Method**: `POST`
*   **Headers**: `Content-Type: application/json`
*   **Request Body**:
    ```json
    {
      "github_url": "https://github.com/username/repository"
    }
    ```
*   **Example Response**:
    ```json
    {
      "repository": "repository",
      "iq_score": 85.2,
      "maintainability_risk": 15.4,
      "technical_debt_score": 12.0,
      "architecture_quality": 88.5,
      "repograph_score": 86.8
    }
    ```

---

## 🧪 Testing

Run tests to ensure everything is set up correctly:

```bash
pytest tests/
```
