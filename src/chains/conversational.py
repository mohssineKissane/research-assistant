"""
Conversational QA Chain - Multi-Turn Conversations with Memory
===============================================================

This module extends the basic RetrievalQA to support multi-turn conversations.
It remembers previous questions and answers, enabling follow-up questions.

How Conversational QA Differs from Basic QA:
    Basic QA: Each question is independent
    - User: "What is AI?"
    - Bot: "AI is..."
    - User: "Give examples" ❌ Bot doesn't know what to give examples of
    
    Conversational QA: Questions build on previous context
    - User: "What is AI?"
    - Bot: "AI is..."
    - User: "Give examples" ✅ Bot knows to give examples of AI

LangChain's ConversationalRetrievalChain:
    1. Takes question + chat history
    2. Reformulates question using history (standalone question)
    3. Retrieves relevant documents
    4. Generates answer considering both documents and history
    5. Saves Q&A pair to memory
"""
from langchain.chains import ConversationalRetrievalChain
from src.utils.prompts import PromptTemplates


class ConversationalQAChain:
    """
    Wrapper for LangChain's ConversationalRetrievalChain.
    
    This enables multi-turn conversations where the assistant:
    - Remembers previous questions and answers
    - Understands follow-up questions with pronouns ("it", "that", "this")
    - Maintains conversation context across multiple exchanges
    """
    
    def __init__(self, llm, vectorstore, memory):
        """
        Initialize conversational QA chain.
        
        Args:
            llm: Language model for generation (e.g., ChatGroq)
            vectorstore: ChromaDB instance with embedded documents
            memory: ConversationMemoryManager instance
                   Stores chat history and provides it to the chain
        """
        self.llm = llm
        self.vectorstore = vectorstore
        self.memory = memory
        self.chain = None
    
    def create_chain(self, k=4):
        """
        Create the conversational retrieval chain.
        
        This method builds a chain that:
        1. Takes a question + chat history
        2. Reformulates the question to be standalone (resolves pronouns)
        3. Retrieves relevant documents
        4. Generates answer using both documents and conversation context
        
        Args:
            k: Number of document chunks to retrieve per question
               - More chunks = better context but higher token cost
               - Typical: 3-6 chunks
               
        How Question Reformulation Works:
            User: "What is machine learning?"
            Bot: "Machine learning is..."
            User: "What are its applications?"
            
            Without reformulation: Searches for "its applications" ❌
            With reformulation: Searches for "machine learning applications" ✅
            
            The chain uses an LLM to rewrite follow-up questions into
            standalone questions that can be searched independently.
        """
        
        # Create retriever from vectorstore
        # This searches for similar document chunks using embeddings
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        # Create ConversationalRetrievalChain
        # This is LangChain's built-in chain for conversational QA
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,                        # LLM for both reformulation and answer generation
            retriever=retriever,                 # Retrieves relevant document chunks
            memory=self.memory.get_memory(),     # Conversation memory for context
            return_source_documents=True,        # Include source docs for citations
            verbose=True,                        # Shows internal steps (useful for debugging)
            combine_docs_chain_kwargs={
                # Custom prompt that works with chat history
                "prompt": PromptTemplates.get_conversational_prompt()
            }
        )
        
        return self.chain
    
    def ask(self, question: str):
        """
        Ask a question with conversation context.
        
        This method:
        1. Loads chat history from memory
        2. Reformulates question if it's a follow-up (resolves pronouns)
        3. Retrieves relevant documents
        4. Generates answer considering both documents and history
        5. Automatically saves Q&A to memory
        
        Args:
            question: User's question (can be a follow-up with pronouns)
            
        Returns:
            dict with:
            - "answer": Generated response
            - "sources": Source documents used
            - "chat_history": Full conversation history
            
        Example Multi-Turn Conversation:
            >>> chain.ask("What is neural networks?")
            {"answer": "Neural networks are...", ...}
            
            >>> chain.ask("How do they learn?")  # "they" = neural networks
            {"answer": "They learn through...", ...}
            
        The chain automatically understands "they" refers to neural networks
        because it has access to the conversation history.
        """
        # Call the chain with the question
        # The chain automatically:
        # 1. Loads chat_history from memory
        # 2. Reformulates question if needed
        # 3. Retrieves documents
        # 4. Generates answer
        # 5. Saves to memory
        result = self.chain({"question": question})
        
        return {
            "answer": result["answer"],              # Generated response
            "sources": result["source_documents"],   # Documents used for answer
            "chat_history": self.memory.get_history()  # Full conversation
        }
    
    def reset_conversation(self):
        """
        Start a new conversation by clearing memory.
        
        Use this when:
        - Switching to a different topic
        - Starting a fresh research session
        - The conversation context is no longer relevant
        
        After calling this, the next question will be treated as
        the start of a new conversation with no prior context.
        
        Example:
            >>> chain.ask("What is AI?")
            >>> chain.ask("Tell me more")  # Remembers "AI"
            >>> chain.reset_conversation()
            >>> chain.ask("Tell me more")  # Doesn't know what to tell more about
        """
        self.memory.clear()