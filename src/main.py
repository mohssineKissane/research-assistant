"""
Main Research Assistant - The Orchestrator of the RAG Pipeline
================================================================

This is the main entry point for the RAG (Retrieval-Augmented Generation) system.
It coordinates all components: document processing, vector storage, and question-answering.

RAG Flow Overview:
    1. Load PDFs → Split into chunks → Convert to vectors → Store in ChromaDB
    2. User asks question → Find similar chunks → Send to LLM with context → Return answer

Usage:
    from src.main import ResearchAssistant
    
    assistant = ResearchAssistant()
    assistant.load_documents(["path/to/document.pdf"])
    assistant.setup_qa()
    result = assistant.ask_question("What is this document about?")
"""
from typing import List
from src.utils.config import config
from src.utils.llm import llm_manager
from src.processing.document_processing_pipeline import DocumentProcessingPipeline
from src.chains.retrieval_qa import RetrievalQAChain
from src.chains.conversational import ConversationalQAChain
from src.memory.conversation_memory import ConversationMemoryManager
from src.utils.formatters import ResponseFormatter

# Agent imports for autonomous decision-making
from src.agent.research_agent import ResearchAgent
from src.agent.agent_config import AgentConfig


class ResearchAssistant:
    """
    Main class that orchestrates the entire RAG pipeline.
    
    This class ties together:
    - DocumentProcessingPipeline: Loads, splits, and indexes documents
    - RetrievalQAChain: Handles question-answering with retrieval
    - ResponseFormatter: Formats answers with citations
    - ResearchAgent: Autonomous agent that selects tools to answer questions
    
    The RAG pattern allows the LLM to answer questions based on YOUR documents,
    not just its training data. This prevents hallucination and grounds answers in facts.
    """
    
    def __init__(self):
        """
        Initialize the Research Assistant.
        
        Sets up:
        - Config: Loaded from config.yaml (chunk size, model names, etc.)
        - Pipeline: Ready to process documents (but no docs loaded yet)
        - VectorStore: None until documents are loaded
        - QA Chain: None until setup_qa() is called
        - Agent: None until setup_agent() is called
        """
        from src.processing.document_processing_pipeline import PipelineConfig
        
        # Load configuration from config.yaml
        self.config = config
        
        # Create pipeline config with settings from YAML
        # PipelineConfig holds: chunk_size, chunk_overlap, embedding_model, vectorstore_path
        pipeline_config = PipelineConfig.from_yaml()
        
        # Initialize the document processing pipeline
        # This creates: DocumentLoader, DocumentSplitter, EmbeddingsGenerator, ChromaVectorStore
        self.pipeline = DocumentProcessingPipeline(pipeline_config)
        
        # These will be set when documents are loaded and QA is set up
        self.vectorstore = None  # ChromaDB instance (set by load_documents)
        self.qa_chain = None     # RetrievalQAChain instance (set by setup_qa)
        
        # Conversational capabilities
        self.conversational_chain = None  # ConversationalQAChain (set by setup_conversational_qa)
        self.memory = None                # ConversationMemoryManager (set by setup_conversational_qa)
        
        # Agent components - for autonomous tool selection and decision-making
        self.agent = None                 # ResearchAgent instance (set by setup_agent)
        self.agent_config = AgentConfig() # Default agent configuration
    
    def load_documents(self, pdf_paths: List[str]):
        """
        Load and process PDF documents into the vector store.
        
        This is the "indexing" phase of RAG:
        1. Load PDFs → Each page becomes a LangChain Document object
        2. Split into chunks → Large pages are split into ~1000 char pieces
        3. Generate embeddings → Each chunk becomes a 384-dimensional vector
        4. Store in ChromaDB → Vectors are indexed for fast similarity search
        
        Args:
            pdf_paths: List of paths to PDF files to load
            
        Example:
            assistant.load_documents(["data/samples/sample.pdf"])
        
        After calling this:
        - self.vectorstore contains the ChromaDB instance with all document vectors
        - You can search documents or set up QA
        """
        print("📥 Processing documents...")
        
        # process_pdfs does: Load → Split → Embed → Store
        # Returns the ChromaDB vectorstore instance
        self.vectorstore = self.pipeline.process_pdfs(pdf_paths)
        
        print("✅ Documents loaded and indexed")
    
    def setup_qa(self, k=4):
        """
        Initialize the QA (Question-Answering) chain.
        
        This creates the retrieval-augmented generation chain:
        - Retriever: Fetches top-k most similar document chunks for each question
        - LLM: Generates answers based on retrieved context
        - Prompt: Instructs the LLM how to answer (cite sources, stay grounded)
        
        Args:
            k: Number of document chunks to retrieve for each question.
               More chunks = more context but higher token cost.
               Typical values: 3-6
        
        LangChain's RetrievalQA chain handles:
        1. Embed the user's question (using same model as documents)
        2. Find k most similar chunks in ChromaDB
        3. Combine chunks into a context string
        4. Fill prompt template with context + question
        5. Send to LLM and get answer
        6. Return answer + source documents
        
        Must call load_documents() first!
        """
        if self.vectorstore is None:
            raise ValueError("No documents loaded. Call load_documents() first")
        
        # Get LLM instance (ChatGroq with llama-3.3-70b-versatile)
        llm = llm_manager.get_llm()
        
        # Create the QA chain wrapper
        # This wraps LangChain's RetrievalQA with our custom prompt
        self.qa_chain = RetrievalQAChain(llm, self.vectorstore)
        
        # Initialize the chain with k chunks to retrieve
        # This creates the retriever and connects everything
        self.qa_chain.create_chain(k=k)
        
        print(f"✅ QA system ready (retrieving top {k} chunks)")
    
    def ask_question(self, question: str) -> dict:
        """
        Ask a question and get an answer with source citations.
        
        This is the "query" phase of RAG:
        1. Your question is embedded into a vector
        2. ChromaDB finds the k most similar document chunks
        3. Chunks are combined into a context string
        4. LLM receives: context + question + instructions
        5. LLM generates an answer citing the sources
        6. Answer is formatted with citations
        
        Args:
            question: Natural language question about your documents
            
        Returns:
            dict with:
            - 'answer': The LLM's response
            - 'citations': List of source documents with page numbers
            - 'num_sources': Count of unique sources used
            
        Example:
            result = assistant.ask_question("What is AI?")
            print(result['answer'])
            print(result['citations'])
        
        Must call setup_qa() first!
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Call setup_qa() first")
        
        # Ask the question through our QA chain
        # Internally: embed question → search → combine context → call LLM
        result = self.qa_chain.ask(question)
        
        # Format the response with deduplicated citations
        formatted = ResponseFormatter.format_answer_with_sources(
            result['answer'],   # The LLM's generated answer
            result['sources']   # List of LangChain Document objects used as context
        )
        
        return formatted
    
    def ask_and_display(self, question: str):
        """
        Ask a question and print a nicely formatted output.
        
        Convenience method that:
        1. Calls ask_question()
        2. Prints formatted answer with sources
        3. Returns the result dict
        
        Good for interactive use in notebooks or terminal.
        """
        result = self.ask_question(question)
        print(ResponseFormatter.format_for_display(result))
        return result
    
    # ==================== Conversational QA Methods ====================
    
    def setup_conversational_qa(self, k=4, memory_type="buffer_window", memory_k=5):
        """
        Initialize the conversational QA chain with memory.
        
        This enables multi-turn conversations where the assistant:
        - Remembers previous questions and answers
        - Understands follow-up questions with pronouns
        - Maintains context across the conversation
        
        Args:
            k: Number of document chunks to retrieve per question (default: 4)
            memory_type: Type of conversation memory to use
                - "buffer": Store all messages (unlimited)
                - "buffer_window": Store last N exchanges (recommended)
            memory_k: Number of recent exchanges to remember (for buffer_window)
                      Default: 5 (remembers last 5 Q&A pairs)
        
        Example:
            >>> assistant.load_documents(["paper.pdf"])
            >>> assistant.setup_conversational_qa()
            >>> assistant.ask_conversational("What is this paper about?")
            >>> assistant.ask_conversational("Tell me more about it")  # "it" = the paper
        
        Difference from setup_qa():
            - setup_qa(): Each question is independent
            - setup_conversational_qa(): Questions build on previous context
        
        Must call load_documents() first!
        """
        if self.vectorstore is None:
            raise ValueError("No documents loaded. Call load_documents() first")
        
        # Create conversation memory manager
        # This stores chat history and provides it to the chain
        self.memory = ConversationMemoryManager(
            memory_type=memory_type,
            k=memory_k
        )
        
        # Get LLM instance (ChatGroq)
        llm = llm_manager.get_llm()
        
        # Create conversational QA chain
        # This wraps LangChain's ConversationalRetrievalChain
        self.conversational_chain = ConversationalQAChain(
            llm=llm,
            vectorstore=self.vectorstore,
            memory=self.memory
        )
        
        # Initialize the chain
        self.conversational_chain.create_chain(k=k)
        
        print(f"✅ Conversational QA ready (memory: {memory_type}, k={memory_k})")
    
    def ask_conversational(self, question: str) -> dict:
        """
        Ask a question in a conversational context.
        
        This method supports multi-turn conversations:
        - Follow-up questions with pronouns ("it", "that", "this")
        - References to previous topics
        - Contextual understanding across exchanges
        
        The chain automatically:
        1. Loads conversation history from memory
        2. Reformulates the question to be standalone
        3. Retrieves relevant documents
        4. Generates answer using both documents and history
        5. Saves the Q&A to memory
        
        Args:
            question: Natural language question (can reference previous context)
            
        Returns:
            dict with:
            - 'answer': The LLM's response
            - 'citations': List of source documents with page numbers
            - 'num_sources': Count of unique sources used
            - 'chat_history': List of previous Q&A pairs
            
        Example Conversation:
            >>> result = assistant.ask_conversational("What is neural networks?")
            >>> print(result['answer'])
            "Neural networks are..."
            
            >>> result = assistant.ask_conversational("How do they learn?")
            >>> print(result['answer'])
            "They learn through backpropagation..."  # Understands "they" = neural networks
        
        Must call setup_conversational_qa() first!
        """
        if self.conversational_chain is None:
            raise ValueError("Conversational chain not initialized. Call setup_conversational_qa() first")
        
        # Ask the question through the conversational chain
        # This handles question reformulation, retrieval, and memory
        result = self.conversational_chain.ask(question)
        
        # Format the response with citations
        formatted = ResponseFormatter.format_answer_with_sources(
            result['answer'],
            result['sources']
        )
        
        # Add chat history to the result
        formatted['chat_history'] = result['chat_history']
        
        return formatted
    
    def ask_conversational_and_display(self, question: str):
        """
        Ask a conversational question and print formatted output.
        
        Convenience method for interactive use that:
        1. Calls ask_conversational()
        2. Prints formatted answer with sources
        3. Shows conversation history
        4. Returns the result dict
        
        Good for notebooks and terminal sessions.
        """
        result = self.ask_conversational(question)
        
        # Print the formatted answer
        print(ResponseFormatter.format_for_display(result))
        
        # Optionally show conversation history
        if result.get('chat_history'):
            print("\n📜 Conversation History:")
            for i, msg in enumerate(result['chat_history'], 1):
                role = "👤 User" if msg['role'] == 'user' else "🤖 Assistant"
                print(f"{role}: {msg['content'][:100]}...")  # Show first 100 chars
        
        return result
    
    def reset_conversation(self):
        """
        Clear conversation memory and start fresh.
        
        Use this when:
        - Switching to a different topic
        - Starting a new research session
        - The conversation context is no longer relevant
        
        After calling this, the next question will be treated as
        the start of a new conversation with no prior context.
        
        Example:
            >>> assistant.ask_conversational("What is AI?")
            >>> assistant.ask_conversational("Tell me more")  # Remembers "AI"
            >>> assistant.reset_conversation()
            >>> assistant.ask_conversational("Tell me more")  # Doesn't know what to tell more about
        """
        if self.memory is None:
            print("⚠️ No conversation memory to reset")
            return
        
        self.memory.clear()
        print("✅ Conversation memory cleared")
    
    def get_conversation_history(self):
        """
        Get the full conversation history.
        
        Returns:
            List of message dicts with 'role' and 'content':
            [
                {'role': 'user', 'content': 'What is AI?'},
                {'role': 'assistant', 'content': 'AI is...'},
                ...
            ]
            
        Useful for:
        - Displaying chat history in UI
        - Exporting conversation transcripts
        - Debugging conversation flow
        """
        if self.memory is None:
            return []
        
        return self.memory.get_history()
    
    # ==================== Agent Methods (Autonomous Decision-Making) ====================
    
    def setup_agent(self):
        """
        Initialize the research agent with autonomous tool selection.
        
        The agent uses the ReAct (Reasoning + Acting) pattern to:
        1. Analyze the user's question
        2. Decide which tool(s) to use (document search, web search, summarization)
        3. Execute the selected tool(s)
        4. Synthesize a final answer
        
        Difference from setup_qa():
            - setup_qa(): Always uses document retrieval (fixed approach)
            - setup_agent(): Autonomously decides which tools to use (flexible approach)
        
        The agent can:
        - Search documents when needed
        - Search the web for current information
        - Summarize content when requested
        - Combine multiple tools in a single query
        
        Example Agent Reasoning:
            User: "What does the paper say about AI, and what's new in 2024?"
            
            Thought: Need document info first, then current events
            Action 1: search_documents("AI")
            Observation 1: [Found AI info in paper]
            
            Thought: Now need current 2024 developments
            Action 2: search_web("AI developments 2024")
            Observation 2: [Found recent articles]
            
            Final Answer: [Combines both sources]
        
        Must call load_documents() first!
        
        Returns:
            The initialized agent executor
        """
        if self.vectorstore is None:
            raise ValueError("No documents loaded. Call load_documents() first")
        
        # Get LLM with agent-appropriate temperature
        # Agent reasoning benefits from slightly higher temperature for flexibility
        llm = llm_manager.get_llm(
            temperature=self.agent_config.temperature
        )
        
        # Get all agent configuration (verbose, max_iterations, prompts, etc.)
        # The config class now automatically handles default prompts if needed
        agent_kwargs = self.agent_config.get_agent_kwargs()
        
        # Create the research agent with tools
        research_agent = ResearchAgent(llm, self.vectorstore)
        
        # Initialize the agent executor
        # This creates the ReAct loop (Thought → Action → Observation)
        # We pass **agent_kwargs to inject config like max_iterations and custom prompts
        self.agent = research_agent.create_agent(
            agent_type=self.agent_config.agent_type,
            **agent_kwargs
        )
        
        print("✓ Research agent ready")
        print(f"  Agent type: {self.agent_config.agent_type}")
        print(f"  Tools available: document_search, web_search, summarize_content")
        print(f"  Verbose mode: {self.agent_config.verbose}")
        
        return self.agent
    
    def setup_agent_with_memory(self, memory_type="buffer_window", memory_k=5):
        """
        Initialize the research agent with conversation memory.
        
        This creates an agent that:
        - Remembers previous questions and answers
        - Understands follow-up questions with pronouns
        - Autonomously selects tools (document search, web search, summarization)
        - Combines the power of agent decision-making with conversational context
        
        Args:
            memory_type: Type of conversation memory to use
                - "buffer": Store all messages (unlimited)
                - "buffer_window": Store last N exchanges (recommended)
            memory_k: Number of recent exchanges to remember (for buffer_window)
                      Default: 5 (remembers last 5 Q&A pairs)
        
        Example:
            >>> assistant.load_documents(["paper.pdf"])
            >>> assistant.setup_agent_with_memory()
            >>> assistant.ask_agent("What is this paper about?")
            >>> assistant.ask_agent("Tell me more about it")  # "it" = the paper
        
        Difference from setup_agent():
            - setup_agent(): Each question is independent (no memory)
            - setup_agent_with_memory(): Questions build on previous context
        
        Difference from setup_conversational_qa():
            - setup_conversational_qa(): Fixed approach (always retrieves from docs)
            - setup_agent_with_memory(): Autonomous tool selection + memory
        
        Must call load_documents() first!
        
        Returns:
            The initialized agent executor with memory
        """
        if self.vectorstore is None:
            raise ValueError("No documents loaded. Call load_documents() first")
        
        # Create conversation memory manager
        # This stores chat history and provides it to the agent
        agent_memory = ConversationMemoryManager(
            memory_type=memory_type,
            k=memory_k
        )
        
        # Get LLM with agent-appropriate temperature
        llm = llm_manager.get_llm(
            temperature=self.agent_config.temperature
        )
        
        # Get all agent configuration
        agent_kwargs = self.agent_config.get_agent_kwargs()
        
        # Create the research agent with tools AND memory
        research_agent = ResearchAgent(llm, self.vectorstore, memory=agent_memory)
        
        # Initialize the agent executor
        # The agent will automatically use conversational mode because memory is provided
        self.agent = research_agent.create_agent(
            agent_type=self.agent_config.agent_type,
            **agent_kwargs
        )
        
        print("✓ Research agent with memory ready")
        print(f"  Agent type: conversational-react-description (auto-selected)")
        print(f"  Tools available: document_search, web_search, summarize_content")
        print(f"  Memory: {memory_type}, k={memory_k}")
        print(f"  Verbose mode: {self.agent_config.verbose}")
        
        return self.agent
    
    def ask_agent(self, query: str) -> str:
        """
        Ask the agent a question - agent autonomously decides which tools to use.
        
        Unlike ask_question() which always uses document retrieval,
        the agent analyzes the query and chooses appropriate tools:
        - Document-specific questions → Uses document_search
        - Current events questions → Uses web_search
        - Summary requests → Uses summarize_content
        - Complex questions → May use multiple tools
        
        The agent's decision-making process (when verbose=True):
            > Entering new AgentExecutor chain...
            Thought: I need to search the documents for this information
            Action: search_documents
            Action Input: "transformers architecture"
            Observation: [Document search results]
            Thought: I now have enough information to answer
            Final Answer: [Synthesized answer]
        
        Args:
            query: Natural language question or instruction
            
        Returns:
            String containing the agent's final answer
            
        Example:
            >>> assistant.setup_agent()
            >>> answer = assistant.ask_agent("What does the paper say about transformers?")
            >>> print(answer)
            "According to the paper, transformers are..."
            
            >>> answer = assistant.ask_agent("Summarize all documents")
            >>> print(answer)
            "The documents cover the following topics..."
        
        Must call setup_agent() first!
        """
        if self.agent is None:
            raise ValueError("Agent not initialized. Call setup_agent() first")
        
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}\n")
        
        # Run the agent - it will autonomously select and use tools
        # The ReAct loop continues until the agent has a final answer
        result = self.agent.run(query)
        
        print(f"\n{'='*60}")
        print(f"FINAL ANSWER:")
        print(result)
        print(f"{'='*60}\n")
        
        return result