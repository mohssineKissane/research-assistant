# Personal Research Assistant with Multi-Source Analysis

A powerful assistant that takes a research question, searches multiple sources (PDFs, web), synthesizes information, and produces a structured report with citations.

## 🚀 Core Functionality

- **Multi-Source Analysis**: Combine information from uploaded documents (PDFs) and live web searches.
- **Natural Language Q&A**: Ask questions about your documents in plain English.
- **Smart Citations**: Get answers with precise source citations (document name, page number).
- **Contextual Memory**: Have multi-turn conversations where the assistant remembers previous details.
- **Automated Reporting**: Export your research findings as a structured Markdown report.
- **Intelligent Agent**: Automatically decides when to use local documents vs. searching the web for broader context.

## 🛠️ Tech Stack

This project is built using a completely free and open-source stack:

| Component | Technology | Description |
|-----------|------------|-------------|
| **LLM** | Groq (mixtral-8x7b-32768) | High-performance open-source model |
| **Embeddings** | HuggingFace (sentence-transformers) | Local semantic search capabilities |
| **Vector DB** | Chroma | Local vector storage |
| **Web Search** | DuckDuckGo | Public web search without API keys |
| **Framework** | LangChain | Orchestration of chains and agents |
| **UI** | Streamlit | Interactive web interface |
| **Doc Processing** | PyPDF2 | PDF text extraction |

## 🏗️ Architecture

```
User (Streamlit UI)
    ↓
Research Agent (decides which tool to use)
    ↓
    ├─→ Document Search Tool → Vector Store (Chroma)
    ├─→ Web Search Tool → DuckDuckGo API
    └─→ Summarization Tool
    ↓
Synthesis Chain (combines sources)
    ↓
LLM (Groq) generates answer
    ↓
Conversation Memory (tracks context)
    ↓
Response with citations → User
```

## 📂 Project Structure

```
research-assistant/
├── src/
│   ├── agent/              # Research agent with ReAct logic
│   ├── tools/              # Tools for doc search, web search, summarization
│   ├── chains/             # QA, conversational, and synthesis chains
│   ├── processing/         # Document loading, splitting, and embedding
│   ├── vectorstore/        # Chroma DB operations
│   ├── memory/             # Conversation context management
│   └── utils/              # Helper functions and configurations
├── app/
│   └── streamlit_app.py    # Main UI application
├── data/
│   ├── uploaded/           # Directory for user uploaded PDFs
│   └── vectorstore/        # Persisted Vector DB
├── tests/                  # Unit tests
├── .env                    # Environment variables (API keys)
├── pyproject.toml          # Project metadata and dependencies (uv)
└── uv.lock                 # Locked dependency versions
```

## 💡 Example Usage

1. **Upload**: User uploads research papers (PDFs) via the Streamlit UI.
2. **Indexer**: The system processes and indexes documents for semantic search.
3. **Query**: User asks, "What are the main findings regarding X?"
4. **Retrieval**: The agent searches documents and provides an answer with citations.
5. **Deep Dive**: User asks about recent developments. The agent detects the need for external info, searches the web, and synthesizes it with document data.
6. **Export**: User downloads a complete research report containing the entire session's findings.

## 📦 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/research-assistant.git
   cd research-assistant
   ```

2. **Install uv** (if not already installed)
   ```bash
   pip install uv
   ```

3. **Install dependencies**
   ```bash
   uv sync
   ```
   This will create a `.venv` virtual environment with Python 3.11 and install all dependencies from `pyproject.toml`.

4. **Configure API Keys**
   Create a `.env` file and add your keys (e.g., GROQ_API_KEY).

5. **Run the App**
   ```bash
   uv run streamlit run app/streamlit_app.py
   ```
   Or activate the environment first:
   ```bash
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   
   streamlit run app/streamlit_app.py
   ```

## ⚠️ Current Limitations

- **Authentication**: Single-user local instance only.
- **File Support**: Currently supports PDF files only.
- **Deployment**: Designed for local execution.

---
*Note: This is an initial version of the project documentation.*
