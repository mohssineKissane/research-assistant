"""
State Manager - Centralized Streamlit Session State Management
================================================================

This module manages Streamlit's session state to maintain data across reruns.

Why Session State?
    Streamlit reruns the entire script on every interaction. Without session state,
    all variables would reset. Session state persists data between reruns.

What We Store:
    - ResearchAssistant instance (expensive to recreate)
    - Current session ID and all sessions
    - Uploaded files and processing status
    - Chat history for display
    - User settings (k, memory type, etc.)
"""
import streamlit as st
from src.main import ResearchAssistant
from datetime import datetime
import uuid


def initialize_session_state():
    """
    Initialize all session state variables.
    
    This function is called once when the app first loads.
    It sets up default values for all state variables.
    
    Why check 'initialized'?
        Streamlit reruns the script on every interaction. We only want
        to initialize once, not on every rerun.
    """
    if 'initialized' not in st.session_state:
        # Core assistant instance
        # This is expensive to create, so we only do it once
        st.session_state.assistant = ResearchAssistant()
        
        # Session management
        # Each session is a separate conversation with its own memory
        st.session_state.current_session_id = str(uuid.uuid4())
        st.session_state.sessions = {
            st.session_state.current_session_id: {
                'name': 'Session 1',
                'created_at': datetime.now(),
                'messages': []  # Chat history for this session
            }
        }
        
        # Document management
        st.session_state.uploaded_files = []  # List of uploaded file names
        st.session_state.documents_processed = False  # Whether docs are indexed
        st.session_state.processing_status = ""  # Status message
        
        # Chat state
        st.session_state.chat_history = []  # Current session's messages
        st.session_state.waiting_for_response = False  # Loading state
        
        # Settings
        # These control how the assistant works
        st.session_state.settings = {
            'mode': 'simple',                    # 'simple' or 'agent'
            'k': 4,                          # Number of chunks to retrieve
            'memory_type': 'buffer_window',  # Type of conversation memory
            'memory_k': 5,                   # Number of exchanges to remember
            'show_sources': True,            # Display source citations
            'show_timestamps': True          # Show message timestamps
        }
        
        # Mode-specific state
        st.session_state.agent_initialized = False  # Whether agent mode is set up
        
        # UI state
        st.session_state.current_tab = 'Chat'  # Active tab
        
        # Mark as initialized
        st.session_state.initialized = True


def get_current_session():
    """
    Get the current session data.
    
    Returns:
        dict: Current session with name, created_at, and messages
    """
    return st.session_state.sessions[st.session_state.current_session_id]


def create_new_session():
    """
    Create a new conversation session.
    
    This:
    - Generates a new session ID
    - Creates session data structure
    - Switches to the new session
    - Resets the assistant's conversation memory
    
    Use when:
    - User wants to start a fresh conversation
    - Switching to a different topic
    """
    # Generate new session ID
    new_session_id = str(uuid.uuid4())
    
    # Calculate session number for naming
    session_num = len(st.session_state.sessions) + 1
    
    # Create session data
    st.session_state.sessions[new_session_id] = {
        'name': f'Session {session_num}',
        'created_at': datetime.now(),
        'messages': []
    }
    
    # Switch to new session
    st.session_state.current_session_id = new_session_id
    st.session_state.chat_history = []
    
    # Reset assistant's conversation memory if it exists
    if hasattr(st.session_state.assistant, 'reset_conversation'):
        st.session_state.assistant.reset_conversation()
    
    return new_session_id


def switch_session(session_id):
    """
    Switch to a different session.
    
    Args:
        session_id: UUID of the session to switch to
        
    This:
    - Changes current session
    - Loads that session's chat history
    - Updates assistant's memory (future enhancement)
    """
    if session_id in st.session_state.sessions:
        st.session_state.current_session_id = session_id
        st.session_state.chat_history = st.session_state.sessions[session_id]['messages']
        
        # Reset assistant's memory
        # Note: In future, we could restore the session's memory
        if hasattr(st.session_state.assistant, 'reset_conversation'):
            st.session_state.assistant.reset_conversation()


def clear_current_session():
    """
    Clear the current session's messages.
    
    This:
    - Removes all messages from current session
    - Resets chat history
    - Clears assistant's conversation memory
    
    Use when:
    - User wants to clear the conversation but keep the session
    """
    current_session = get_current_session()
    current_session['messages'] = []
    st.session_state.chat_history = []
    
    # Reset assistant's memory
    if hasattr(st.session_state.assistant, 'reset_conversation'):
        st.session_state.assistant.reset_conversation()


def add_message(role, content, sources=None):
    """
    Add a message to the current session.
    
    Args:
        role: "user" or "assistant"
        content: Message text
        sources: Optional list of source documents (for assistant messages)
        
    This stores the message in:
    - Current session's messages (for persistence)
    - Chat history (for display)
    """
    message = {
        'role': role,
        'content': content,
        'timestamp': datetime.now(),
        'sources': sources or []
    }
    
    # Add to current session
    current_session = get_current_session()
    current_session['messages'].append(message)
    
    # Add to chat history for display
    st.session_state.chat_history.append(message)


def get_session_list():
    """
    Get list of all sessions for display.
    
    Returns:
        list: List of tuples (session_id, session_name, created_at)
    """
    sessions = []
    for session_id, session_data in st.session_state.sessions.items():
        sessions.append((
            session_id,
            session_data['name'],
            session_data['created_at']
        ))
    
    # Sort by creation time (newest first)
    sessions.sort(key=lambda x: x[2], reverse=True)
    
    return sessions


def update_settings(key, value):
    """
    Update a setting value.
    
    Args:
        key: Setting name (e.g., 'k', 'memory_type')
        value: New value
    """
    st.session_state.settings[key] = value
