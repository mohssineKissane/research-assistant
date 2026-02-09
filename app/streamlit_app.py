"""
Research Assistant - Streamlit Web Interface
=============================================

Main entry point for the Streamlit application.

This app provides:
- PDF document upload and processing
- Conversational Q&A with memory
- Session management
- Source citation display
- Conversation export

Usage:
    streamlit run app/streamlit_app.py
"""
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import components and utilities
from app.utils.state_manager import initialize_session_state
from app.utils.ui_helpers import apply_custom_css
from app.components.sidebar import render_sidebar
from app.components.chat_interface import render_chat_interface
from app.components.document_viewer import render_document_viewer
from app.components.history_viewer import render_history_viewer


def main():
    """
    Main application entry point.
    
    This sets up the page configuration, initializes state,
    and renders the UI components.
    """
    # Page configuration
    st.set_page_config(
        page_title="Research Assistant",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Apply custom CSS
    apply_custom_css()
    
    # Render sidebar
    render_sidebar()
    
    # Main content area with tabs
    render_main_content()


def render_main_content():
    """
    Render the main content area with tabs.
    
    Tabs:
    - Chat: Main conversational interface
    - Documents: View uploaded documents
    - History: View and export conversation history
    """
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Documents", "📜 History"])
    
    # Chat tab
    with tab1:
        render_chat_interface()
    
    # Documents tab
    with tab2:
        render_document_viewer()
    
    # History tab
    with tab3:
        render_history_viewer()


# Run the app
if __name__ == "__main__":
    main()
