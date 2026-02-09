"""
History Viewer Component - Conversation History and Export
===========================================================

This module provides conversation history viewing and export functionality.
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from app.utils.ui_helpers import export_conversation, format_timestamp
from app.utils.state_manager import get_current_session


def render_history_viewer():
    """
    Render the history viewer tab.
    
    Features:
    - Display full conversation history
    - Export as Markdown
    - Download button
    """
    st.header("📜 History")
    
    current_session = get_current_session()
    messages = current_session['messages']
    
    if not messages:
        st.info("No conversation history yet. Start chatting to see history here!")
        return
    
    # Session info
    st.subheader(f"{current_session['name']}")
    st.caption(f"Created: {format_timestamp(current_session['created_at'])}")
    st.caption(f"Messages: {len(messages)}")
    
    # Export button
    if st.button("📥 Export Conversation"):
        markdown_content = export_conversation(messages, current_session['name'])
        
        st.download_button(
            label="Download as Markdown",
            data=markdown_content,
            file_name=f"{current_session['name'].replace(' ', '_')}.md",
            mime="text/markdown"
        )
    
    st.markdown("---")
    
    # Display full history
    st.subheader("Full Conversation")
    
    for i, msg in enumerate(messages, 1):
        role = "👤 You" if msg['role'] == 'user' else "🤖 Assistant"
        timestamp = format_timestamp(msg.get('timestamp'))
        
        with st.expander(f"{i}. {role} - {timestamp}", expanded=False):
            st.write(msg['content'])
            
            # Show sources if available
            sources = msg.get('sources', [])
            if sources:
                st.markdown("**Sources:**")
                for source in sources:
                    filename = source.metadata.get('filename', 'Unknown')
                    page = source.metadata.get('page', 'N/A')
                    st.text(f"- {filename} (Page {page})")
