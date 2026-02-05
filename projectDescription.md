# Personal Research Assistant with Multi-Source Analysis

System that takes a research question, searches multiple sources (PDFs, web), synthesizes information, and produces a structured report with citations.

## Project Details

### 1. Project Goal
Build a **Personal Research Assistant** in **1 week** to learn LangChain by implementing all its core concepts through a practical, working application.

---

### 2. What It Does

**Core Functionality:**
- Upload multiple PDF documents
- Ask questions about the documents in natural language
- Get answers with source citations (document name, page number)
- Have multi-turn conversations with context memory
- Automatically search the web when documents don't have enough info
- Combine information from multiple sources (documents + web)
- Export research findings as a report

**Example Use Case:**
User uploads 3 research papers → asks "What are the main findings?" → gets synthesized answer with citations → asks follow-ups → agent decides to search web for recent info → combines all sources → exports report

---

### 3. Tech Stack (100% FREE)

| Component | Technology | Why |
|-----------|------------|-----|
| **LLM** | Groq (mixtral-8x7b-32768) | Free, super fast |
| **Embeddings** | HuggingFace (sentence-transformers) | Free, runs locally |
| **Vector DB** | Chroma | Free, local, simple |
| **Web Search** | DuckDuckGo | Free, no API key |
| **Framework** | LangChain | Core learning goal |
| **UI** | Streamlit | Fast Python UI |
| **Doc Processing** | PyPDF2 | PDF parsing |
| **Language** | Python 3.11 | Managed with uv |

**Total Cost: $0**

---

### 4. Architecture

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

---

### 5. Project Structure

```
research-assistant/
├── src/
│   ├── agent/              # Research agent with ReAct
│   ├── tools/              # Document search, web search, summarization
│   ├── chains/             # QA, conversational, synthesis chains
│   ├── processing/         # Document loader, text splitter, embeddings
│   ├── vectorstore/        # Chroma operations
│   ├── memory/             # Conversation memory
│   └── utils/              # Prompts, formatters, config
├── app/
│   └── streamlit_app.py    # UI
├── data/
│   ├── uploaded/           # User PDFs
│   ├── samples/            # Test PDFs
│   └── vectorstore/        # Chroma DB
├── notebooks/              # Jupyter experiments
├── tests/                  # Unit tests
├── .env                    # API keys
├── config.yaml             # Configuration
├── pyproject.toml          # Project metadata and dependencies (uv)
└── uv.lock                 # Locked dependency versions
```

---


### 6. Development Timeline

| Day | Focus | Deliverable |
|-----|-------|-------------|
| **1** | Document processing | Load PDFs into vector DB |
| **2** | Basic Q&A | Ask questions, get cited answers |
| **3** | Conversation | Multi-turn chat with memory |
| **4** | Agent | Smart tool selection |
| **5** | Multi-source | Web search + synthesis |
| **6** | UI & Reports | Streamlit app + export |
| **7** | Polish | Testing, docs, demo |

**Time commitment:** 6-8 hours/day focused work

---

### 7. Key Features (MVP)

**Must Have:**
- [x] Upload multiple PDFs
- [x] Semantic search with citations
- [x] Conversational Q&A
- [x] Agent decides doc vs web search
- [x] Multi-source synthesis
- [x] Streamlit chat UI
- [x] Export report (Markdown)

**Nice to Have (if time):**
- [ ] PDF export
- [ ] Document summaries
- [ ] Advanced error handling
- [ ] Performance optimization

---

### 9. User Journey

```
1. User opens Streamlit app
2. Uploads 3 PDF research papers
3. System processes & indexes (30 seconds)
4. User asks: "What are the main findings?"
5. Agent searches documents, returns answer with citations
6. User asks: "What about recent 2024 developments?"
7. Agent searches web, combines with document info
8. User asks 3-4 follow-up questions
9. System maintains conversation context
10. User clicks "Export Report"
11. Downloads Markdown file with all Q&A and citations
```

**Total time: ~5 minutes** (vs hours of manual reading)

---

### 10. Success Criteria

**By end of Week 1:**
- ✅ Working Streamlit app
- ✅ Can process 5+ PDFs
- ✅ Accurate answers with citations
- ✅ Agent correctly chooses tools
- ✅ Conversation memory works
- ✅ Web search integration functional
- ✅ Can export research report
- ✅ Code is clean and documented
- ✅ Demo-ready for portfolio

---

### 11. What We're NOT Building

❌ User authentication
❌ Multi-user support
❌ Cloud deployment
❌ Advanced UI/UX design
❌ Mobile responsiveness
❌ Multiple document formats (Word, Excel, etc.)
❌ Production-grade error handling
❌ Extensive test coverage
❌ Performance optimization
