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
    3. Sets up the appropriate mode (Simple or Agent)
    4. Updates state
    """
    try:
        # Use an absolute path so it works on both local and Streamlit Cloud
        temp_dir = Path(__file__).parent.parent.parent / "data" / "temp_uploads"
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
        
        # Get current mode and settings
        settings = st.session_state.settings
        mode = settings['mode']
        
        # Setup based on selected mode
        if mode == 'simple':
            # Simple Mode: Conversational QA (always retrieves from documents)
            with st.spinner("🧠 Setting up Simple Mode (Conversational QA)..."):
                st.session_state.assistant.setup_conversational_qa(
                    k=settings['k'],
                    memory_type=settings['memory_type'],
                    memory_k=settings['memory_k']
                )
                st.session_state.agent_initialized = False
        else:
            # Agent Mode: Autonomous research agent with memory
            with st.spinner("🤖 Setting up Agent Mode (Autonomous Research)..."):
                st.session_state.assistant.setup_agent_with_memory(
                    memory_type=settings['memory_type'],
                    memory_k=settings['memory_k']
                )
                st.session_state.agent_initialized = True
        
        # Update state
        st.session_state.uploaded_files = [f.name for f in uploaded_files]
        st.session_state.documents_processed = True
        st.session_state.processing_status = f"Documents processed in {mode.upper()} mode"
        
        st.success(f"✅ Documents processed and ready in {mode.upper()} mode!")
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
    - Mode selection (Simple vs Agent)
    - Number of chunks to retrieve (k)
    - Memory type
    - Memory window size
    - Display options
    """
    with st.expander("⚙️ Settings", expanded=False):
        settings = st.session_state.settings
        
        # Mode Selection
        st.write("**🎯 Mode Selection**")
        
        mode = st.radio(
            "Choose your research mode:",
            options=['simple', 'agent'],
            format_func=lambda x: "🔹 Simple Mode" if x == 'simple' else "🤖 Agent Mode",
            index=0 if settings['mode'] == 'simple' else 1,
            help="Select how the assistant should process your questions"
        )
        
        # Mode descriptions
        if mode == 'simple':
            st.info("""
            **Simple Mode (Conversational QA)**
            - ✅ Fast and predictable
            - ✅ Always searches your documents
            - ✅ Remembers conversation history
            - ✅ Best for: Quick questions about your documents
            - ⚠️ Limited to document content only
            """)
        else:
            st.info("""
            **Agent Mode (Autonomous Research)**
            - 🤖 Intelligent tool selection
            - ✅ Can search documents when needed
            - ✅ Can search the web for current info
            - ✅ Can summarize and combine sources
            - ✅ Remembers conversation history
            - ✅ Best for: Complex research requiring multiple sources
            - ⚠️ Slightly slower due to decision-making
            """)
        
        if mode != settings['mode']:
            update_settings('mode', mode)
            if st.session_state.documents_processed:
                st.warning("⚠️ Mode changed. Please reprocess documents to apply.")
        
        st.markdown("---")
        
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
                    current_mode = settings['mode']
                    
                    if current_mode == 'simple':
                        # Reinitialize conversational QA
                        st.session_state.assistant.setup_conversational_qa(
                            k=settings['k'],
                            memory_type=settings['memory_type'],
                            memory_k=settings['memory_k']
                        )
                        st.session_state.agent_initialized = False
                    else:
                        # Reinitialize agent with memory
                        st.session_state.assistant.setup_agent_with_memory(
                            memory_type=settings['memory_type'],
                            memory_k=settings['memory_k']
                        )
                        st.session_state.agent_initialized = True
                        
                    st.success(f"Settings applied! Using {current_mode.upper()} mode.")
                except Exception as e:
                    st.error(f"Error applying settings: {str(e)}")
            else:
                st.info("Upload and process documents first")
