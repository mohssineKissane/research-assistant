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
from src.utils.formatters import ResponseFormatter


class ResearchAssistant:
    """
    Main class that orchestrates the entire RAG pipeline.
    
    This class ties together:
    - DocumentProcessingPipeline: Loads, splits, and indexes documents
    - RetrievalQAChain: Handles question-answering with retrieval
    - ResponseFormatter: Formats answers with citations
    
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