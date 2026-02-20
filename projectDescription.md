# Research Assistant — Project Description

A personal research assistant that answers questions from uploaded PDFs and the live web. Supports two modes: a lightweight conversational RAG pipeline and a full autonomous agent with tool selection.

🌐 **Live Demo:** [research-assistantgit-mmgpmwqhp9bnhhpzy64yrh.streamlit.app](https://research-assistantgit-mmgpmwqhp9bnhhpzy64yrh.streamlit.app/)

---

## 1. Project Goal

Build a **Personal Research Assistant** to learn LangChain by implementing all its core concepts through a practical, working application:
- RAG (Retrieval-Augmented Generation) pipelines
- Conversational chains with memory
- Autonomous agents with tool selection (ReAct pattern)
- Streamlit UI with session management
- Cloud deployment

---

## 2. Two Modes

### 🔹 Simple Mode — Conversational RAG
A fast, predictable document Q&A chain.

- User uploads PDFs → chunked, embedded, stored in ChromaDB
- Every question retrieves the top-k most relevant chunks
- LangChain `ConversationalRetrievalChain` reformulates follow-up questions using chat history
- Answers always include source citations (file + page)
- Best for: questions where the answer is in the documents

### 🤖 Agent Mode — Autonomous Research (Default)
An autonomous ReAct agent that decides which tool to use.

**ReAct Loop:**
```
Thought → Action (tool call) → Observation → Thought → ... → Final Answer
```

**Available tools:**
| Tool | When the agent uses it |
|------|------------------------|
| `search_documents` | Question is about uploaded PDFs |
| `search_web` | Need current info, news, or info not in docs |
| `summarize_content` | User asks for a summary |

- Agent can chain multiple tools in one query
- Remembers conversation history (follow-up questions understood)
- Best for: complex research, recent events, multi-source synthesis

---

## 3. Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM** | Groq — `llama-3.3-70b-versatile` |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` (local) |
| **Vector DB** | ChromaDB (local) |
| **Web Search** | Tavily API |
| **Framework** | LangChain (chains, agents, memory) |
| **UI** | Streamlit |
| **Doc Processing** | PyPDF |
| **Package Manager** | uv |
| **Deployment** | Streamlit Community Cloud |

---

## 4. Architecture

```
User (Streamlit UI)
    ↓
┌──────────────────────────────────────────┐
│  Simple Mode          Agent Mode         │
│  ConversationalQA     ReAct Agent        │
│  Chain                    ↓              │
│      ↓           ┌────┬──┴──┬────┐       │
│      ↓           ↓    ↓     ↓    ↓       │
│      ↓        DocSearch Web  Summarize   │
└──────┬──────────────┬──────┬──────┬──────┘
       ↓              ↓      ↓      ↓
   ChromaDB      ChromaDB Tavily ChromaDB
       ↓
 LLM (Groq llama-3.3-70b-versatile)
       ↓
 ConversationMemory (BufferWindowMemory)
       ↓
 Response with citations → User
```

---

## 5. Project Structure

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
├── notebooks/              # Jupyter experiments per LangChain concept
├── .streamlit/
│   └── config.toml         # Streamlit server + theme config
├── config.yaml             # LLM, embeddings, vectorstore settings
├── requirements.txt        # Production dependencies
├── pyproject.toml          # Project metadata (uv)
└── .env                    # API keys (local only, never committed)
```

---

## 6. Key Implementation Details

**Memory:** `ConversationBufferWindowMemory` keeps the last 5 exchanges. Uses `output_key="answer"` for Simple Mode and `output_key="output"` for Agent Mode (matching `AgentExecutor`'s output key).

**Agent prompts:** The `conversational-react-description` agent requires a custom prefix/suffix with `{chat_history}`, `{input}`, `{agent_scratchpad}`. The prefix explicitly instructs the agent to always use tools before answering.

**Token management:** Web search results are capped at 3 results, content at 400 chars, URLs at 100 chars to stay within Groq's TPM limits.

**Secrets:** On Streamlit Cloud, `st.secrets` values are injected into `os.environ` at startup so all `os.getenv()` calls work without any code changes.

---

## 7. Feature Checklist

**Completed:**
- [x] Upload and process multiple PDFs
- [x] Semantic search with source citations
- [x] Conversational Q&A with memory (Simple Mode)
- [x] Autonomous agent with tool selection (Agent Mode)
- [x] Live web search via Tavily
- [x] Streamlit chat UI with session management
- [x] Mode switching (Simple ↔ Agent) in settings
- [x] Deployed to Streamlit Community Cloud

**Not in scope:**
- ❌ User authentication / multi-user
- ❌ PDF export of reports
- ❌ Non-PDF file formats (Word, Excel)
- ❌ Persistent storage on cloud (re-upload needed after restart)
- ❌ Mobile-optimised UI

---

## 8. Deployment

**Platform:** Streamlit Community Cloud (free tier)

**Live URL:** https://research-assistantgit-mmgpmwqhp9bnhhpzy64yrh.streamlit.app/

**Required secrets (set in App Settings → Secrets):**
```toml
GROQ_API_KEY = "..."
TAVILY_API_KEY = "..."
```

**Note:** No persistent disk on Streamlit Cloud — uploaded PDFs and the ChromaDB vector store reset on each cold restart. Users must re-upload documents per session.
