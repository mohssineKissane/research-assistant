"""
Document Viewer Component - Display Uploaded Documents
=======================================================

This module provides a view of uploaded documents and their metadata.
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))


def render_document_viewer():
    """
    Render the document viewer tab.
    
    Shows:
    - List of uploaded documents
    - Document metadata
    - All sources used in current session
    """
    st.header("📄 Documents")
    
    if not st.session_state.uploaded_files:
        st.info("No documents uploaded yet. Use the sidebar to upload PDFs.")
        return
    
    # Display uploaded documents
    st.subheader("Uploaded Documents")
    for i, filename in enumerate(st.session_state.uploaded_files, 1):
        with st.expander(f"📄 {filename}", expanded=False):
            st.write(f"**File:** {filename}")
            st.write(f"**Status:** Processed ✅")
    
    # Display sources from current session
    st.markdown("---")
    st.subheader("Sources Used in This Session")
    
    # Collect all sources from chat history
    all_sources = []
    for message in st.session_state.chat_history:
        if message['role'] == 'assistant' and message.get('sources'):
            all_sources.extend(message['sources'])
    
    if all_sources:
        st.write(f"**Total sources:** {len(all_sources)}")
        for i, source in enumerate(all_sources, 1):
            filename = source.metadata.get('filename', 'Unknown')
            page = source.metadata.get('page', 'N/A')
            st.text(f"{i}. {filename} (Page {page})")
    else:
        st.info("No sources yet. Start asking questions!")
