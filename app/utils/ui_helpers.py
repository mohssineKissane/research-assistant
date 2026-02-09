"""
UI Helpers - Formatting and Display Utilities
==============================================

This module provides helper functions for formatting and displaying
UI elements in the Streamlit app.

Functions:
    - display_message(): Format chat messages
    - display_sources(): Show source citations
    - format_timestamp(): Format datetime objects
    - export_conversation(): Export chat as Markdown
    - apply_custom_css(): Load custom styling
"""
import streamlit as st
from datetime import datetime
from typing import List, Dict


def display_message(message: Dict, show_timestamp: bool = True):
    """
    Display a chat message with proper formatting.
    
    Args:
        message: Message dict with 'role', 'content', 'timestamp', 'sources'
        show_timestamp: Whether to show the timestamp
        
    This creates a nicely formatted message bubble with:
    - Different styling for user vs assistant
    - Optional timestamp
    - Source citations (for assistant messages)
    """
    role = message['role']
    content = message['content']
    timestamp = message.get('timestamp')
    sources = message.get('sources', [])
    
    # Create columns for message alignment
    if role == 'user':
        # User messages: right-aligned
        col1, col2 = st.columns([1, 3])
        with col2:
            st.markdown(f"""
            <div style="background-color: #E3F2FD; padding: 15px; border-radius: 10px; margin: 10px 0;">
                <div style="color: #1976D2; font-weight: bold;">👤 You</div>
                <div style="margin-top: 5px; color: #212121;">{content}</div>
                {f'<div style="font-size: 0.8em; color: #666; margin-top: 5px;">{format_timestamp(timestamp)}</div>' if show_timestamp and timestamp else ''}
            </div>
            """, unsafe_allow_html=True)
    else:
        # Assistant messages: left-aligned
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"""
            <div style="background-color: #F5F5F5; padding: 15px; border-radius: 10px; margin: 10px 0;">
                <div style="color: #424242; font-weight: bold;">🤖 Assistant</div>
                <div style="margin-top: 5px; color: #212121;">{content}</div>
                {f'<div style="font-size: 0.8em; color: #666; margin-top: 5px;">{format_timestamp(timestamp)}</div>' if show_timestamp and timestamp else ''}
            </div>
            """, unsafe_allow_html=True)
            
            # Display sources if available
            if sources:
                display_sources(sources)


def display_sources(sources: List[Dict]):
    """
    Display source citations as expandable cards.
    
    Args:
        sources: List of source document dicts with metadata
        
    Each source shows:
    - Document filename
    - Page number
    - Relevant excerpt (if available)
    """
    if not sources:
        return
    
    with st.expander(f"📎 Sources ({len(sources)} documents)", expanded=False):
        for i, source in enumerate(sources, 1):
            # Extract metadata from source document
            filename = source.metadata.get('filename', 'Unknown')
            page = source.metadata.get('page', 'N/A')
            
            st.markdown(f"""
            **Source {i}:** {filename} (Page {page})
            """)
            
            # Show excerpt if available
            if hasattr(source, 'page_content'):
                excerpt = source.page_content[:200] + "..." if len(source.page_content) > 200 else source.page_content
                st.text(excerpt)
            
            if i < len(sources):
                st.divider()


def format_timestamp(timestamp: datetime) -> str:
    """
    Format a datetime object for display.
    
    Args:
        timestamp: datetime object
        
    Returns:
        str: Formatted time string (e.g., "2:30 PM")
    """
    if not timestamp:
        return ""
    
    # If today, show time only
    if timestamp.date() == datetime.now().date():
        return timestamp.strftime("%I:%M %p")
    
    # If this year, show month and day
    if timestamp.year == datetime.now().year:
        return timestamp.strftime("%b %d, %I:%M %p")
    
    # Otherwise, show full date
    return timestamp.strftime("%b %d %Y, %I:%M %p")


def export_conversation(messages: List[Dict], session_name: str = "Conversation") -> str:
    """
    Export conversation as Markdown.
    
    Args:
        messages: List of message dicts
        session_name: Name of the session
        
    Returns:
        str: Markdown-formatted conversation
    """
    markdown = f"# {session_name}\n\n"
    markdown += f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    markdown += "---\n\n"
    
    for msg in messages:
        role = "**You:**" if msg['role'] == 'user' else "**Assistant:**"
        timestamp = format_timestamp(msg.get('timestamp'))
        content = msg['content']
        
        markdown += f"{role} _{timestamp}_\n\n"
        markdown += f"{content}\n\n"
        
        # Add sources if available
        sources = msg.get('sources', [])
        if sources:
            markdown += "**Sources:**\n"
            for i, source in enumerate(sources, 1):
                filename = source.metadata.get('filename', 'Unknown')
                page = source.metadata.get('page', 'N/A')
                markdown += f"- {filename} (Page {page})\n"
            markdown += "\n"
        
        markdown += "---\n\n"
    
    return markdown


def apply_custom_css():
    """
    Apply custom CSS styling to the Streamlit app.
    
    This improves the appearance of:
    - Chat message bubbles
    - Buttons
    - Sidebar
    - Overall layout
    """
    st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        font-weight: 500;
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed #ccc;
        border-radius: 5px;
        padding: 1rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 500;
        color: #1976D2;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Scrollable chat area */
    .element-container {
        max-height: none;
    }
    
    /* Make chat messages scrollable */
    section[data-testid="stVerticalBlock"] {
        max-height: 600px;
        overflow-y: auto;
        padding-right: 10px;
    }
    
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }
    </style>
    """, unsafe_allow_html=True)


def show_loading_message():
    """
    Display a loading message while processing.
    
    Returns:
        Streamlit spinner context manager
    """
    return st.spinner("🤔 Thinking...")


def show_success_message(message: str):
    """Display a success message."""
    st.success(f"✅ {message}")


def show_error_message(message: str):
    """Display an error message."""
    st.error(f"❌ {message}")


def show_info_message(message: str):
    """Display an info message."""
    st.info(f"ℹ️ {message}")


def show_warning_message(message: str):
    """Display a warning message."""
    st.warning(f"⚠️ {message}")
