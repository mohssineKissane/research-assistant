# Research Assistant

A personal research assistant that answers questions from uploaded PDFs and the live web. Built with LangChain, Groq, and Streamlit.

🌐 **Live Demo:** [research-assistantgit-mmgpmwqhp9bnhhpzy64yrh.streamlit.app](https://research-assistantgit-mmgpmwqhp9bnhhpzy64yrh.streamlit.app/)

---

## 🎯 Two Modes

### 🔹 Simple Mode (RAG)
Fast, document-focused conversational Q&A.
- Always retrieves from your uploaded PDFs
- Answers with source citations (document + page number)
- Remembers conversation history
- Best for: questions where the answer is in your documents

### 🤖 Agent Mode (RAG + Web + Summarization) — Default
Autonomous research agent using the ReAct pattern (Reasoning + Acting).
- Decides which tool to use based on the question
- Searches uploaded PDFs when the answer is there
- Searches the live web (Tavily) for current events or missing information
- Summarizes document content on request
- Remembers conversation history
- Best for: complex research requiring multiple sources or up-to-date information

---

## 🚀 Features

- **PDF upload & indexing** — Upload one or more PDFs; they are chunked, embedded, and indexed in ChromaDB
- **Semantic search** — Questions are matched to the most relevant document chunks
- **Source citations** — Every document answer includes the source file and page number
- **Conversational memory** — Multi-turn chat; the assistant understands follow-up questions
- **Live web search** — Tavily API fetches current information the documents don't contain
- **Session management** — Create and switch between multiple conversation sessions
- **Streamlit UI** — Clean chat interface with sidebar settings

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM** | Groq — `llama-3.3-70b-versatile` |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` (runs locally) |
| **Vector DB** | ChromaDB (local) |
| **Web Search** | Tavily API |
| **Framework** | LangChain (chains, agents, memory) |
| **UI** | Streamlit |
| **Doc Processing** | PyPDF |
| **Package Manager** | uv |

---

## 🏗️ Architecture

```
User (Streamlit UI)
    ↓
┌─────────────────────────────────────────┐
│  Simple Mode          Agent Mode        │
│  (Conversational      (ReAct Agent)     │
│   RAG Chain)              ↓             │
│       ↓           ┌──────┴──────┐       │
│       │           ↓      ↓      ↓       │
│       │      Doc Search  Web  Summarize │
└───────┼───────────┼──────┼──────┼───────┘
        ↓           ↓      ↓      ↓
    ChromaDB    ChromaDB Tavily ChromaDB
        ↓
  LLM (Groq llama-3.3-70b-versatile)
        ↓
  Conversation Memory
        ↓
  Response with citations → User
```

---

## 📂 Project Structure

```
research-assistant/
├── src/
│   ├── agent/              # ResearchAgent (ReAct) + AgentConfig
│   ├── tools/              # document_search, web_search, summarization
│   ├── chains/             # conversational.py, retrieval_qa.py
│   ├── processing/         # PDF loader, text splitter, embeddings pipeline
│   ├── vectorstore/        # ChromaDB wrapper
│   ├── memory/             # ConversationMemoryManager
│   └── utils/              # config, llm, prompts, formatters
├── app/
│   ├── streamlit_app.py    # Main entry point
│   ├── components/         # chat_interface, sidebar, document_viewer
│   └── utils/              # state_manager, ui_helpers
├── data/
│   ├── temp_uploads/       # Uploaded PDFs (session)
│   └── vectorstore/        # ChromaDB index
├── notebooks/              # Jupyter experiments
├── .streamlit/
│   └── config.toml         # Streamlit server + theme config
├── config.yaml             # LLM, embeddings, vectorstore settings
├── requirements.txt        # Production dependencies
├── pyproject.toml          # Project metadata (uv)
└── .env                    # API keys (local only, not committed)
```

---

## 📦 Local Setup

### Prerequisites
- Python 3.11
- [uv](https://github.com/astral-sh/uv) package manager
- [Groq API key](https://console.groq.com/keys) (free)
- [Tavily API key](https://tavily.com) (free — 1000 searches/month)

### Steps

```bash
# 1. Clone
git clone https://github.com/yourusername/research-assistant.git
cd research-assistant

# 2. Install dependencies
uv sync

# 3. Create .env
echo GROQ_API_KEY=your_key_here >> .env
echo TAVILY_API_KEY=your_key_here >> .env

# 4. Run
uv run streamlit run app/streamlit_app.py
```

---

## ☁️ Deployment

Deployed on **Streamlit Community Cloud**.

🌐 [research-assistantgit-mmgpmwqhp9bnhhpzy64yrh.streamlit.app](https://research-assistantgit-mmgpmwqhp9bnhhpzy64yrh.streamlit.app/)

To deploy your own instance:
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Set main file: `app/streamlit_app.py`
4. Add secrets (`GROQ_API_KEY`, `TAVILY_API_KEY`) in App Settings → Secrets

> **Note:** Streamlit Cloud has no persistent storage. Uploaded PDFs and the vector store reset on each restart — users need to re-upload documents per session.

---

## ⚠️ Limitations

- PDF files only (no Word, Excel, etc.)
- Single-user (no authentication)
- No persistent storage on Streamlit Cloud (re-upload needed after restart)
- Groq free tier has rate limits (TPM/RPM)
