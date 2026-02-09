
# Streamlit Frontend - User Guide

## 🚀 Quick Start

### Running the App

```bash
# Activate virtual environment
.venv\Scripts\activate

# Run Streamlit app
streamlit run app/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📖 How to Use

### 1. Upload Documents

1. Click **"Browse files"** in the sidebar
2. Select one or more PDF files
3. Click **"🔄 Process Documents"**
4. Wait for processing to complete (you'll see ✅ when ready)

### 2. Ask Questions

1. Type your question in the input box at the bottom
2. Click **"📤 Send"**
3. The assistant will respond with an answer and sources

### 3. Follow-Up Questions

The assistant remembers your conversation! You can ask follow-up questions:

**Example:**
- You: "What is machine learning?"
- Assistant: "Machine learning is..."
- You: "What are its applications?" ← The assistant knows "its" = machine learning

### 4. Manage Sessions

**Start New Conversation:**
- Click **"➕ New Conversation"** in the sidebar
- This creates a fresh session with empty memory

**Switch Sessions:**
- Click on any previous session name to switch back
- Each session has its own conversation history

**Clear Current Session:**
- Click **"🗑️ Clear Current Session"**
- This removes all messages but keeps the session

### 5. Adjust Settings

In the sidebar **Settings** section:

- **Chunks to retrieve**: How many document pieces to search (1-10)
  - More = better context, but slower
  - Recommended: 4-6

- **Memory type**: How conversation history is stored
  - **buffer_window**: Keeps last N exchanges (recommended)
  - **buffer**: Keeps all exchanges

- **Memory window size**: Number of Q&A pairs to remember (1-10)
  - Recommended: 5

- **Show sources**: Toggle source citations on/off
- **Show timestamps**: Toggle message timestamps on/off

Click **"💾 Apply Settings"** to update (requires documents to be processed)

### 6. View Documents

Click the **"📄 Documents"** tab to:
- See all uploaded documents
- View all sources used in the current session

### 7. Export Conversation

Click the **"📜 History"** tab to:
- View full conversation history
- Export as Markdown file
- Download for later reference

---

## 💡 Tips

1. **Better Questions = Better Answers**
   - Be specific
   - Reference document topics
   - Use follow-up questions to dig deeper

2. **Memory Management**
   - The assistant remembers recent exchanges (default: last 5)
   - For new topics, start a new session
   - Clear session if context becomes confusing

3. **Source Citations**
   - Click on source expandable sections to see which documents were used
   - Sources show filename and page number
   - Use this to verify information

4. **Multiple Documents**
   - You can upload multiple PDFs at once
   - The assistant searches across all documents
   - Sources will indicate which document the answer came from

---

## 🐛 Troubleshooting

### "No documents loaded" error
- Make sure you clicked "🔄 Process Documents" after uploading
- Wait for the ✅ success message before asking questions

### Slow responses
- Reduce "Chunks to retrieve" in settings
- Check your internet connection (LLM is cloud-based)

### Memory not working
- Make sure you're using conversational QA (not basic QA)
- Check memory settings in sidebar
- Try starting a new session

### Sources not showing
- Enable "Show sources" in settings
- Click "💾 Apply Settings"

---

## 📁 File Structure

```
app/
├── streamlit_app.py           # Main entry point
├── components/
│   ├── sidebar.py            # Document upload & settings
│   ├── chat_interface.py     # Chat UI
│   ├── document_viewer.py    # Document display
│   └── history_viewer.py     # History & export
└── utils/
    ├── state_manager.py      # Session state
    └── ui_helpers.py         # UI formatting
```

---

## 🎯 Features

✅ PDF document upload and processing
✅ Conversational Q&A with memory
✅ Multi-session management
✅ Source citation display
✅ Conversation export (Markdown)
✅ Customizable settings
✅ Beautiful UI with message bubbles
✅ Timestamps and metadata

---

## 🔧 Advanced

### Custom Styling

The app uses custom CSS for styling. To modify:
- Edit `app/utils/ui_helpers.py`
- Look for the `apply_custom_css()` function
- Modify the CSS in the `st.markdown()` call

### Adding Features

The modular structure makes it easy to add features:
- New components go in `app/components/`
- Shared utilities go in `app/utils/`
- Update `streamlit_app.py` to integrate

---

## 📝 Notes

- Session state persists during app session only (not across restarts)
- Uploaded files are stored temporarily in `data/temp_uploads/`
- Conversation history is in-memory (export to save permanently)
- The app uses your existing `ResearchAssistant` backend

---

**Enjoy your Research Assistant! 🔬**
