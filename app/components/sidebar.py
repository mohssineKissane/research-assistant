"""
Sidebar Component - Document Upload, Session Management, and Settings
======================================================================

This module provides the sidebar UI for the Streamlit app.

Sections:
    1. Document Upload: Upload and process PDFs
    2. Session Management: Create, switch, and clear sessions
    3. Settings: Configure retrieval and memory parameters
"""
import streamlit as st
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.utils.state_manager import (
    create_new_session,
    switch_session,
    clear_current_session,
    get_session_list,
    get_current_session,
    update_settings
)
from app.utils.ui_helpers import (
    show_success_message,
    show_error_message,
    show_info_message
)


def render_sidebar():
    """
    Render the complete sidebar with all sections.
    
    This is the main function called from the app to display:
    - Document upload
    - Session management
    - Settings panel
    """
    with st.sidebar:
        st.title("🔬 Research Assistant")
        st.markdown("---")
        
        # Section 1: Document Upload
        render_document_upload()
        
        st.markdown("---")
        
        # Section 2: Session Management
        render_session_management()
        
        st.markdown("---")
        
        # Section 3: Settings
        render_settings()


def render_document_upload():
    """
    Render the document upload section.
    
    Features:
    - File uploader for PDFs
    - List of uploaded documents
    - Process button
    - Status indicators
    """
    st.subheader("📁 Document Upload")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF files to analyze"
    )
    
    # Show uploaded files
    if uploaded_files:
        st.write(f"**Uploaded:** {len(uploaded_files)} file(s)")
        for file in uploaded_files:
            st.text(f"📄 {file.name}")
        
        # Process button
        if st.button("🔄 Process Documents", type="primary"):
            process_documents(uploaded_files)
    
    # Show current status
    if st.session_state.documents_processed:
        st.success("✅ Documents ready")
        st.caption(f"{len(st.session_state.uploaded_files)} document(s) indexed")
    elif st.session_state.processing_status:
        st.info(st.session_state.processing_status)


def process_documents(uploaded_files):
    """
    Process uploaded PDF documents.
    
    Args:
        uploaded_files: List of UploadedFile objects from Streamlit
        
    This:
    1. Saves files temporarily
    2. Loads them into the assistant
    3. Sets up conversational QA
    4. Updates state
    """
    try:
        # Create temp directory for uploads
        temp_dir = Path("data/temp_uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded files
        file_paths = []
        with st.spinner("📥 Saving files..."):
            for uploaded_file in uploaded_files:
                file_path = temp_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(str(file_path))
        
        # Load documents into assistant
        with st.spinner("🔍 Processing documents..."):
            st.session_state.assistant.load_documents(file_paths)
        
        # Setup conversational QA
        with st.spinner("🧠 Setting up conversational QA..."):
            settings = st.session_state.settings
            st.session_state.assistant.setup_conversational_qa(
                k=settings['k'],
                memory_type=settings['memory_type'],
                memory_k=settings['memory_k']
            )
        
        # Update state
        st.session_state.uploaded_files = [f.name for f in uploaded_files]
        st.session_state.documents_processed = True
        st.session_state.processing_status = "Documents processed successfully"
        
        st.success("✅ Documents processed and ready for questions!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error processing documents: {str(e)}")
        st.session_state.documents_processed = False
        st.session_state.processing_status = f"Error: {str(e)}"


def render_session_management():
    """
    Render the session management section.
    
    Features:
    - Current session display
    - New session button
    - Session switcher
    - Clear session button
    """
    st.subheader("💬 Sessions")
    
    # Current session info
    current_session = get_current_session()
    st.write(f"**Current:** {current_session['name']}")
    st.caption(f"{len(current_session['messages'])} messages")
    
    # New session button
    if st.button("➕ New Conversation"):
        new_id = create_new_session()
        st.success("New conversation started!")
        st.rerun()
    
    # Session switcher
    sessions = get_session_list()
    if len(sessions) > 1:
        st.write("**Switch to:**")
        for session_id, session_name, created_at in sessions:
            if session_id != st.session_state.current_session_id:
                if st.button(f"📝 {session_name}", key=f"switch_{session_id}"):
                    switch_session(session_id)
                    st.rerun()
    
    # Clear session button
    if len(current_session['messages']) > 0:
        if st.button("🗑️ Clear Current Session", type="secondary"):
            clear_current_session()
            st.success("Session cleared!")
            st.rerun()


def render_settings():
    """
    Render the settings section inside a collapsible expander.
    
    Settings:
    - Number of chunks to retrieve (k)
    - Memory type
    - Memory window size
    - Display options
    """
    with st.expander("⚙️ Settings", expanded=False):
        settings = st.session_state.settings
        
        # Retrieval settings
        st.write("**Retrieval**")
        k = st.slider(
            "Chunks to retrieve",
            min_value=1,
            max_value=10,
            value=settings['k'],
            help="Number of document chunks to retrieve per question"
        )
        if k != settings['k']:
            update_settings('k', k)
        
        # Memory settings
        st.write("**Memory**")
        memory_type = st.selectbox(
            "Memory type",
            options=['buffer_window', 'buffer'],
            index=0 if settings['memory_type'] == 'buffer_window' else 1,
            help="buffer_window: Keep last N exchanges\nbuffer: Keep all exchanges"
        )
        if memory_type != settings['memory_type']:
            update_settings('memory_type', memory_type)
        
        if memory_type == 'buffer_window':
            memory_k = st.slider(
                "Memory window size",
                min_value=1,
                max_value=10,
                value=settings['memory_k'],
                help="Number of recent Q&A pairs to remember"
            )
            if memory_k != settings['memory_k']:
                update_settings('memory_k', memory_k)
        
        # Display settings
        st.write("**Display**")
        show_sources = st.checkbox(
            "Show sources",
            value=settings['show_sources'],
            help="Display source citations with answers"
        )
        if show_sources != settings['show_sources']:
            update_settings('show_sources', show_sources)
        
        show_timestamps = st.checkbox(
            "Show timestamps",
            value=settings['show_timestamps'],
            help="Display message timestamps"
        )
        if show_timestamps != settings['show_timestamps']:
            update_settings('show_timestamps', show_timestamps)
        
        # Apply settings button
        if st.button("💾 Apply Settings"):
            if st.session_state.documents_processed:
                try:
                    # Reinitialize conversational QA with new settings
                    st.session_state.assistant.setup_conversational_qa(
                        k=settings['k'],
                        memory_type=settings['memory_type'],
                        memory_k=settings['memory_k']
                    )
                    st.success("Settings applied!")
                except Exception as e:
                    st.error(f"Error applying settings: {str(e)}")
            else:
                st.info("Upload and process documents first")
