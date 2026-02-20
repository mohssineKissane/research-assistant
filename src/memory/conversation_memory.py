"""
Conversation Memory - Tracking Chat History for Context
========================================================

This module manages conversation history to enable multi-turn dialogues.
It allows the assistant to remember previous questions and answers,
enabling follow-up questions and contextual conversations.

Why Conversation Memory Matters:
    Without memory, each question is isolated:
    - User: "What is AI?"
    - Bot: "AI is..."
    - User: "Tell me more about it"  ❌ Bot doesn't know what "it" refers to
    
    With memory, the bot remembers context:
    - User: "What is AI?"
    - Bot: "AI is..."
    - User: "Tell me more about it"  ✅ Bot knows "it" = AI

LangChain Memory Types:
    - ConversationBufferMemory: Stores all messages (simple, can get large)
    - ConversationBufferWindowMemory: Stores last N messages (memory limit)
    - ConversationSummaryMemory: Summarizes old messages (token efficient)
    
We use BufferWindowMemory to balance context and token usage.
"""
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory


class ConversationMemoryManager:
    """
    Manages conversation history for multi-turn dialogues.
    
    This class wraps LangChain's memory components to provide:
    - Storage of question-answer pairs
    - Retrieval of conversation history
    - Memory clearing for new conversations
    
    The memory is used by ConversationalRetrievalChain to understand
    follow-up questions in context of previous exchanges.
    """
    
    def __init__(self, memory_type="buffer_window", k=5, output_key="answer"):
        """
        Initialize conversation memory.
        
        Args:
            memory_type: Type of memory to use
                - "buffer": Store all messages (unlimited)
                - "buffer_window": Store last k messages (recommended)
            k: Number of recent message pairs to remember (for buffer_window)
               - k=3: Remember last 3 Q&A pairs (6 messages)
               - k=5: Remember last 5 Q&A pairs (10 messages) - default
            output_key: Which output key to save to memory.
                - "answer": for ConversationalRetrievalChain (default)
                - "output": for AgentExecutor (ReAct agents)
               
        Why buffer_window?
            Long conversations can exceed token limits. By keeping only
            recent messages, we maintain context while controlling costs.
        """
        self.memory_type = memory_type
        self.k = k
        
        # Create the appropriate memory type
        # LangChain memory stores messages and provides them to chains
        if memory_type == "buffer":
            # Unlimited memory - stores everything
            # Good for: Short conversations, debugging
            # Risk: Can exceed token limits in long conversations
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",      # Key used in prompts
                return_messages=True,            # Return as Message objects (not strings)
                output_key=output_key            # Which chain output to store
            )
        else:
            # Window memory - stores last k exchanges
            # Good for: Production use, long conversations
            # Benefit: Bounded memory size, predictable token usage
            self.memory = ConversationBufferWindowMemory(
                k=k,                             # Number of exchanges to remember
                memory_key="chat_history",       # Key used in prompts
                return_messages=True,            # Return as Message objects
                output_key=output_key            # Which chain output to store
            )
    
    def get_memory(self):
        """
        Get the LangChain memory object.
        
        Returns:
            LangChain memory instance (ConversationBufferMemory or WindowMemory)
            
        This is passed to ConversationalRetrievalChain which automatically:
        - Saves each Q&A pair to memory after generation
        - Loads chat history before processing new questions
        """
        return self.memory
    
    def get_history(self):
        """
        Get the conversation history as a list of messages.
        
        Returns:
            List of message dicts with 'role' and 'content':
            [
                {'role': 'user', 'content': 'What is AI?'},
                {'role': 'assistant', 'content': 'AI is...'},
                {'role': 'user', 'content': 'Tell me more'},
                {'role': 'assistant', 'content': 'Sure...'}
            ]
            
        Useful for:
        - Displaying chat history in UI
        - Debugging conversation flow
        - Exporting conversation transcripts
        """
        # Load variables from memory
        # chat_history contains the stored messages
        history = self.memory.load_memory_variables({})
        
        # Convert LangChain Message objects to simple dicts
        messages = []
        if "chat_history" in history:
            for msg in history["chat_history"]:
                # LangChain messages have .type (human/ai) and .content
                messages.append({
                    "role": "user" if msg.type == "human" else "assistant",
                    "content": msg.content
                })
        
        return messages
    
    def clear(self):
        """
        Clear all conversation history.
        
        Use this to:
        - Start a fresh conversation
        - Reset context when switching topics
        - Clear memory after exporting a conversation
        
        After clearing, the next question will be treated as a new conversation
        with no prior context.
        """
        self.memory.clear()
    
    def add_exchange(self, question: str, answer: str):
        """
        Manually add a question-answer pair to memory.
        
        Args:
            question: User's question
            answer: Assistant's answer
            
        Note: Usually you don't need to call this manually.
        ConversationalRetrievalChain automatically saves exchanges.
        
        Use this when:
        - Importing conversation history
        - Testing memory behavior
        - Manually constructing conversation context
        """
        # Save context creates a memory entry
        # The chain automatically calls this after each Q&A
        self.memory.save_context(
            {"question": question},  # Input (user message)
            {"answer": answer}       # Output (assistant message)
        )
