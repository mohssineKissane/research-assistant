"""
Chat Interface Component - Conversational Q&A Display
======================================================

This module provides the main chat interface for the Streamlit app.

Features:
    - Display conversation history
    - User and assistant message bubbles
    - Source citations
    - Input box for questions
    - Send button with loading state
"""
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.utils.state_manager import add_message, get_current_session
from app.utils.ui_helpers import display_message, show_loading_message


def render_chat_interface():
    """
    Render the main chat interface.
    
    This displays:
    - Chat history with messages
    - Input box at the bottom
    - Send button
    """
    st.header("💬 Chat")
    
    # Check if documents are processed
    if not st.session_state.documents_processed:
        st.info("👈 Please upload and process documents in the sidebar to start chatting")
        return
    
    # Process any pending question (after loading state is shown)
    process_pending_question()
    
    # Display chat history in a scrollable container
    render_chat_history()
    
    # Input area at bottom
    render_input_area()


def render_chat_history():
    """
    Display all messages in the current session in a scrollable container.
    
    This creates a scrollable area with all messages formatted nicely.
    """
    chat_history = st.session_state.chat_history
    
    if not chat_history:
        st.markdown("""
        <div style="text-align: center; padding: 50px; color: #666;">
            <h3>👋 Welcome!</h3>
            <p>Ask me anything about your documents.</p>
            <p>I'll remember our conversation and can answer follow-up questions.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Create a scrollable container for messages
    # Use a container with custom CSS for scrolling
    chat_container = st.container()
    
    with chat_container:
        # Display each message
        for message in chat_history:
            display_message(
                message,
                show_timestamp=st.session_state.settings['show_timestamps']
            )
        
        # Show loading indicator in chat area if waiting for response
        if st.session_state.waiting_for_response:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("""
                <div style="background-color: #F5F5F5; padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <div style="color: #424242; font-weight: bold;">🤖 Assistant</div>
                    <div style="margin-top: 5px; color: #666;">
                        <i>🤔 Thinking...</i>
                    </div>
                </div>
                """, unsafe_allow_html=True)


def render_input_area():
    """
    Render the input area for asking questions.
    
    Features:
    - Text input box with Enter key support
    - Send button
    - Form submission on Enter or button click
    """
    st.markdown("---")
    
    # Use a form to enable Enter key submission
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_question = st.text_input(
                "Ask a question",
                placeholder="Type your question and press Enter or click Send...",
                label_visibility="collapsed",
                disabled=st.session_state.waiting_for_response,
                key="question_input"
            )
        
        with col2:
            send_button = st.form_submit_button(
                "📤 Send",
                type="primary",
                disabled=st.session_state.waiting_for_response,
                use_container_width=True
            )
    
    # Handle form submission (Enter key or Send button)
    if send_button and user_question:
        handle_user_question(user_question)


def handle_user_question(question: str):
    """
    Handle a user question.
    
    Args:
        question: The user's question text
        
    This:
    1. Adds user message to chat
    2. Sets loading state
    3. Calls the assistant
    4. Adds assistant response to chat
    5. Updates UI
    """
    # Add user message immediately
    add_message('user', question)
    
    # Set loading state
    st.session_state.waiting_for_response = True
    
    # Rerun to show user message and loading indicator
    st.rerun()


# Add a separate function to process the question (called after rerun)
def process_pending_question():
    """
    Process the last user question if waiting for response.
    
    This is called after the UI rerun to show loading state.
    """
    if st.session_state.waiting_for_response:
        # Get the last user message
        chat_history = st.session_state.chat_history
        if chat_history and chat_history[-1]['role'] == 'user':
            question = chat_history[-1]['content']
            
            try:
                # Get response from assistant
                result = st.session_state.assistant.ask_conversational(question)
                
                # Add assistant response
                sources = result.get('sources', []) if st.session_state.settings['show_sources'] else []
                add_message('assistant', result['answer'], sources)
                
            except Exception as e:
                # Handle errors
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                add_message('assistant', error_msg)
            
            finally:
                # Clear loading state
                st.session_state.waiting_for_response = False
                st.rerun()


def render_chat_stats():
    """
    Display chat statistics (optional).
    
    Shows:
    - Number of messages
    - Memory window status
    - Token usage (if available)
    """
    current_session = get_current_session()
    num_messages = len(current_session['messages'])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Messages", num_messages)
    
    with col2:
        memory_k = st.session_state.settings['memory_k']
        memory_status = f"{min(num_messages // 2, memory_k)}/{memory_k}"
        st.metric("Memory", memory_status)
    
    with col3:
        st.metric("Sources", st.session_state.settings['show_sources'] and "On" or "Off")
