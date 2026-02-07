"""
Document Processing Pipeline - Orchestrating the Indexing Phase
================================================================

This module coordinates all document processing steps:
    PDF → Load → Split → Embed → Store

This is the INDEXING phase of RAG - preparing documents for search.
After this pipeline runs, you can search documents semantically.

Pipeline Steps:
    1. DocumentLoader: PDF files → List of Documents (one per page)
    2. DocumentSplitter: Documents → Smaller chunks (for precise retrieval)
    3. EmbeddingsGenerator: Text chunks → Vector representations
    4. ChromaVectorStore: Vectors + metadata → Searchable database

Why a Pipeline?
    - Separates concerns (each step is independent)
    - Easy to modify (swap embedding models, change chunk size)
    - Reusable components (loader can be used standalone)
"""
from typing import List
from langchain_community.vectorstores import Chroma
from langchain.schema import Document


class DocumentProcessingPipeline:
    """
    Complete pipeline for processing documents into a searchable vector store.
    
    This class is the main entry point for the indexing phase.
    It coordinates:
    - Loading PDFs
    - Splitting into chunks
    - Generating embeddings
    - Storing in ChromaDB
    
    All components use the same configuration (from config.yaml).
    """
    
    def __init__(self, config):
        """
        Initialize the pipeline with all components.
        
        Args:
            config: PipelineConfig object with:
                   - chunk_size: How big each text chunk should be
                   - chunk_overlap: Overlap between chunks
                   - embedding_model: Which model to use for embeddings
                   - vectorstore_path: Where to save ChromaDB
        
        Components created:
            - self.loader: DocumentLoader (loads PDFs)
            - self.splitter: DocumentSplitter (chunks text)
            - self.embeddings: EmbeddingsGenerator (text → vectors)
            - self.vectorstore: ChromaVectorStore (stores vectors)
        
        KEY CONNECTION:
            The embeddings generator is passed to the vectorstore.
            The vectorstore saves a reference to it.
            Later, during search, the vectorstore uses this same
            embeddings instance to convert queries to vectors.
        """
        # Import components here to avoid circular imports
        from src.processing.document_loader import DocumentLoader
        from src.processing.text_splitter import DocumentSplitter
        from src.processing.embeddings import EmbeddingsGenerator
        from src.vectorstore.chroma_store import ChromaVectorStore
        
        # Create loader (no config needed - just loads files)
        self.loader = DocumentLoader()
        
        # Create splitter with chunk settings
        self.splitter = DocumentSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
        # Create embeddings generator
        # This instance will be shared with the vectorstore
        self.embeddings = EmbeddingsGenerator(config.embedding_model)
        
        # Create vectorstore wrapper
        # Pass embeddings so it can use them for indexing AND querying
        self.vectorstore = ChromaVectorStore(
            self.embeddings,  # ← This reference is key! Enables semantic search.
            persist_directory=config.vectorstore_path
        )
    
    def process_pdfs(self, file_paths: List[str]) -> Chroma:
        """
        Complete indexing pipeline: Load → Split → Embed → Store
        
        This is the main method you call to index documents.
        After this runs, documents are searchable in ChromaDB.
        
        Args:
            file_paths: List of paths to PDF files
            
        Returns:
            Chroma vectorstore instance (ready for searches)
            
        What happens step by step:
        
        1. LOAD: PDF files → Documents (one per page)
           PyPDFLoader extracts text from each page
           Metadata added: filename, page number, upload date
           
        2. SPLIT: Documents → Chunks
           RecursiveCharacterTextSplitter breaks text into ~1000 char pieces
           Overlap preserves context at chunk boundaries
           Metadata copied to each chunk, chunk_id added
           
        3. EMBED & STORE: Chunks → Vectors → ChromaDB
           Each chunk's text → embedding vector (384 numbers)
           Vector + text + metadata stored in ChromaDB
           Data persisted to disk for later use
        
        Example:
            1 PDF with 10 pages (avg 5000 chars/page)
            → 10 Documents
            → 50 chunks (after splitting)
            → 50 vectors stored in ChromaDB
        """
        # Step 1: Load PDFs
        print(f"📥 Loading {len(file_paths)} PDFs...")
        documents = self.loader.load_multiple_pdfs(file_paths)
        print(f"✅ Loaded {len(documents)} pages")
        
        # Step 2: Split into chunks
        print("✂️  Splitting documents into chunks...")
        chunks = self.splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")
        
        # Step 3: Create vectorstore (embeds automatically)
        # Chroma.from_documents() will:
        # 1. Call embeddings.embed_documents() for all chunks
        # 2. Store vectors + documents in database
        print("🔢 Generating embeddings and storing in vector database...")
        vectorstore = self.vectorstore.create_from_documents(chunks)
        print("✅ Processing complete! Vector store ready for search.")
        
        return vectorstore
    
    def add_more_pdfs(self, file_paths: List[str]):
        """
        Add new documents to existing vectorstore (incremental update).
        
        Use this when you want to add more PDFs without re-indexing everything.
        
        Args:
            file_paths: List of new PDF file paths
        """
        print(f"📥 Loading {len(file_paths)} additional PDFs...")
        documents = self.loader.load_multiple_pdfs(file_paths)
        
        print("✂️  Splitting new documents...")
        chunks = self.splitter.split_documents(documents)
        
        print("🔢 Adding to existing vector store...")
        self.vectorstore.add_documents(chunks)
        print(f"✅ Added {len(chunks)} new chunks to vector store")
    
    def search(self, query: str, k: int = 4):
        """
        Search the vector store for relevant chunks.
        
        This is useful for testing retrieval without the full QA chain.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of k most similar Document chunks
        """
        return self.vectorstore.similarity_search(query, k=k)
    
    def search_with_scores(self, query: str, k: int = 4):
        """
        Search with relevance scores (for debugging).
        
        Returns:
            List of (Document, score) tuples
            Score is cosine similarity (0-1, higher = more similar)
        """
        return self.vectorstore.similarity_search_with_score(query, k=k)


class PipelineConfig:
    """
    Configuration for the document processing pipeline.
    
    Holds all settings needed to set up the pipeline:
    - chunk_size: Max characters per chunk
    - chunk_overlap: Overlap between chunks
    - embedding_model: HuggingFace model name
    - vectorstore_path: Where to save ChromaDB
    
    Can be created from config.yaml using from_yaml() class method.
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        embedding_model: str = None,
        vectorstore_path: str = None
    ):
        """
        Create pipeline config with optional overrides.
        
        Any None values default to config.yaml settings.
        """
        from src.utils.config import config
        
        # Use config values as defaults, allow overrides
        self.chunk_size = chunk_size if chunk_size is not None else config.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else config.chunk_overlap
        self.embedding_model = embedding_model if embedding_model is not None else config.embeddings_model_name
        self.vectorstore_path = vectorstore_path if vectorstore_path is not None else config.vectorstore_persist_directory
    
    @classmethod
    def from_yaml(cls):
        """
        Create PipelineConfig from config.yaml settings.
        
        Convenience method that loads all settings from the YAML file.
        
        Returns:
            PipelineConfig with all settings from config.yaml
        """
        from src.utils.config import config
        
        return cls(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            embedding_model=config.embeddings_model_name,
            vectorstore_path=config.vectorstore_persist_directory
        )