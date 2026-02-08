"""
Session Management - Managing Multiple Conversations
====================================================

This module manages multiple conversation sessions, allowing:
- Multiple concurrent conversations (e.g., different topics)
- Session persistence and retrieval
- Message history tracking per session

Use Cases:
    - Multi-user application: Each user has their own session
    - Topic switching: User can have separate conversations for different topics
    - Session export: Save and restore conversation history

Session Structure:
    Each session contains:
    - memory: ConversationMemoryManager instance
    - created_at: Timestamp when session was created
    - messages: List of all messages with timestamps
"""
import uuid
from datetime import datetime
from src.memory.conversation_memory import ConversationMemoryManager


class SessionManager:
    """
    Manages multiple conversation sessions.
    
    This class allows you to:
    - Create new conversation sessions with unique IDs
    - Track multiple conversations simultaneously
    - Store message history per session
    - Clear or delete sessions
    
    Example Use Case:
        User researching AI and also researching quantum computing.
        They want separate conversations for each topic:
        
        >>> manager = SessionManager()
        >>> ai_session = manager.create_session()
        >>> quantum_session = manager.create_session()
        
        Each session has its own memory and doesn't interfere with the other.
    """
    
    def __init__(self):
        """
        Initialize the session manager.
        
        Creates an empty dictionary to store sessions.
        Each session is identified by a unique UUID.
        """
        # Dictionary mapping session_id -> session data
        # session data: {memory, created_at, messages}
        self.sessions = {}
    
    def create_session(self):
        """
        Create a new conversation session.
        
        Returns:
            str: Unique session ID (UUID)
            
        Each session gets:
        - Unique ID for identification
        - Fresh ConversationMemoryManager (empty history)
        - Creation timestamp
        - Empty message list
        
        Example:
            >>> session_id = manager.create_session()
            >>> print(session_id)
            "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        """
        # Generate unique session ID using UUID4
        # UUID4 is random and virtually guaranteed to be unique
        session_id = str(uuid.uuid4())
        
        # Create session data structure
        self.sessions[session_id] = {
            # Each session gets its own memory manager
            # Uses buffer_window memory to limit token usage
            'memory': ConversationMemoryManager(memory_type="buffer_window"),
            
            # Track when session was created
            'created_at': datetime.now(),
            
            # Store all messages with timestamps for export/display
            'messages': []
        }
        
        return session_id
    
    def get_session(self, session_id):
        """
        Retrieve a session by its ID.
        
        Args:
            session_id: UUID string of the session
            
        Returns:
            dict: Session data or None if not found
            
        Session data contains:
            - memory: ConversationMemoryManager instance
            - created_at: datetime object
            - messages: List of message dicts
        """
        return self.sessions.get(session_id)
    
    def add_message(self, session_id, role, content):
        """
        Add a message to a session's history.
        
        Args:
            session_id: UUID of the session
            role: "user" or "assistant"
            content: Message text
            
        This stores messages separately from the memory manager,
        allowing you to:
        - Export full conversation transcripts
        - Display message history in UI
        - Track exact timestamps of each message
        
        Note: The memory manager also stores messages, but this
        provides an additional record with timestamps.
        """
        if session_id in self.sessions:
            self.sessions[session_id]['messages'].append({
                'role': role,           # "user" or "assistant"
                'content': content,     # Message text
                'timestamp': datetime.now()  # When message was sent
            })
    
    def clear_session(self, session_id):
        """
        Clear a session's memory and messages.
        
        Args:
            session_id: UUID of the session to clear
            
        This:
        - Clears the conversation memory (chat history)
        - Removes all stored messages
        - Keeps the session alive (doesn't delete it)
        
        Use this when:
        - User wants to start fresh in the same session
        - Switching topics within a session
        - Testing conversation flow
        """
        if session_id in self.sessions:
            # Clear LangChain memory (removes chat history)
            self.sessions[session_id]['memory'].clear()
            
            # Clear message list
            self.sessions[session_id]['messages'] = []
    
    def delete_session(self, session_id):
        """
        Permanently delete a session.
        
        Args:
            session_id: UUID of the session to delete
            
        This completely removes the session and all its data.
        Use this when:
        - User logs out (in multi-user app)
        - Session is no longer needed
        - Cleaning up old sessions
        
        After deletion, the session_id is invalid and cannot be used.
        """
        if session_id in self.sessions:
            del self.sessions[session_id]