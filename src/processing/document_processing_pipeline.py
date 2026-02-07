"""
Document Processing Pipeline
Combines all components into a cohesive processing workflow
"""
from typing import List
from langchain_community.vectorstores import Chroma
from langchain.schema import Document


class DocumentProcessingPipeline:
    """
    Complete pipeline for processing documents:
    Load → Split → Embed → Store in Vector Database
    """
    
    def __init__(self, config):
        """
        Initialize pipeline with configuration
        
        Args:
            config: Configuration object with attributes:
                - chunk_size: Size of text chunks
                - chunk_overlap: Overlap between chunks
                - embedding_model: Name of embedding model
                - vectorstore_path: Path to persist vector store
        """
        from src.processing.document_loader import DocumentLoader
        from src.processing.text_splitter import DocumentSplitter
        from src.processing.embeddings import EmbeddingsGenerator
        from src.vectorstore.chroma_store import ChromaVectorStore
        
        self.loader = DocumentLoader()
        self.splitter = DocumentSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        self.embeddings = EmbeddingsGenerator(config.embedding_model)
        self.vectorstore = ChromaVectorStore(
            self.embeddings,
            persist_directory=config.vectorstore_path
        )
    
    def process_pdfs(self, file_paths: List[str]) -> Chroma:
        """
        Complete pipeline: Load → Split → Embed → Store
        
        Args:
            file_paths: List of paths to PDF files
            
        Returns:
            Chroma vectorstore instance
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
        print("🔢 Generating embeddings and storing in vector database...")
        vectorstore = self.vectorstore.create_from_documents(chunks)
        print("✅ Processing complete! Vector store ready for search.")
        
        return vectorstore
    
    def add_more_pdfs(self, file_paths: List[str]):
        """
        Add new documents to existing vectorstore
        
        Args:
            file_paths: List of paths to new PDF files
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
        Search the vector store
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant document chunks
        """
        return self.vectorstore.similarity_search(query, k=k)
    
    def search_with_scores(self, query: str, k: int = 4):
        """
        Search with relevance scores
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        return self.vectorstore.similarity_search_with_score(query, k=k)


class PipelineConfig:
    """Configuration for the document processing pipeline"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vectorstore_path: str = "./data/vectorstore"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.vectorstore_path = vectorstore_path