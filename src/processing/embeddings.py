"""
Embeddings Generator - Converting Text to Vectors
==================================================

This module handles the conversion of text to numerical vectors (embeddings).
Embeddings are the foundation of semantic search in RAG systems.

What are Embeddings?
    - A way to represent text as numbers
    - "What is AI?" → [0.15, -0.32, 0.78, ..., 0.45] (384 numbers)
    - Similar meanings = similar vectors (close in vector space)
    - Enables semantic search, not just keyword matching

Model Used:
    sentence-transformers/all-MiniLM-L6-v2
    - Free, runs locally (no API calls)
    - 384-dimensional vectors
    - Good balance of speed and quality
    - Trained on 1B+ sentence pairs

How Embeddings Enable RAG:
    1. Indexing: Each document chunk → embedding vector → stored in ChromaDB
    2. Query: User question → embedding vector → find similar vectors in DB
    
CRITICAL: Must use the SAME model for indexing and querying!
          If models differ, vectors won't be comparable.
"""
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.utils.config import config


class EmbeddingsGenerator:
    """
    Wrapper around HuggingFace embeddings for text-to-vector conversion.
    
    This class:
    1. Loads a sentence-transformer model
    2. Provides methods to embed queries and documents
    3. Returns the embeddings instance for use with vectorstores
    
    The embeddings object is passed to ChromaDB during setup.
    ChromaDB stores a reference to it, so during search it can
    embed the user's question using the same model.
    """
    
    def __init__(self, model_name=None, device=None, normalize=None):
        """
        Initialize the embeddings generator.
        
        Args:
            model_name: HuggingFace model name
                       Default: "sentence-transformers/all-MiniLM-L6-v2"
                       Other options:
                       - "sentence-transformers/all-mpnet-base-v2" (768 dims, slower)
                       - "sentence-transformers/paraphrase-MiniLM-L6-v2"
                       
            device: "cpu" or "cuda"
                   Use "cuda" if you have a GPU for faster embeddings.
                   
            normalize: Whether to normalize vectors to unit length.
                      Default: True (recommended for cosine similarity)
        
        How the model works:
            1. Text → Tokenizer → Token IDs [101, 2054, 2003, ...]
            2. Token IDs → BERT/Transformer → Hidden states
            3. Hidden states → Mean pooling → Single vector
            4. (Optional) Normalize → Unit vector
        """
        # Use config values as defaults
        # This allows changing settings without modifying code
        if model_name is None:
            model_name = config.embeddings_model_name
        if device is None:
            device = config.embeddings_device
        if normalize is None:
            normalize = config.embeddings_normalize
        
        # Create HuggingFaceEmbeddings instance
        # This is LangChain's wrapper around sentence-transformers
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            
            # Model kwargs passed to the underlying model
            model_kwargs={
                'device': device  # 'cpu' or 'cuda'
            },
            
            # Encoding kwargs control how text is processed
            encode_kwargs={
                'normalize_embeddings': normalize  # L2 normalize for cosine similarity
            }
        )
    
    def get_embeddings(self):
        """
        Return the HuggingFaceEmbeddings instance.
        
        This is passed to ChromaDB so it can embed queries during search.
        
        The flow:
        1. You call: Chroma.from_documents(docs, embedding=self.get_embeddings())
        2. Chroma stores: self._embedding_function = your_embeddings
        3. During search: self._embedding_function.embed_query("question")
        
        This is how the retriever "knows" about your embeddings model -
        it doesn't. It just uses the same object you gave to Chroma.
        """
        return self.embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query/question.
        
        Used during search time to convert user's question to a vector.
        
        Args:
            text: The query text (e.g., "What is AI?")
            
        Returns:
            List of floats: The embedding vector (384 dimensions)
            Example: [0.15, -0.32, 0.78, ..., 0.45]
        
        Note: LangChain's retriever calls this internally through
              vectorstore._embedding_function.embed_query()
        """
        return self.embeddings.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents/chunks.
        
        Used during indexing to convert all document chunks to vectors.
        
        Args:
            texts: List of document chunks
            
        Returns:
            List of embedding vectors (one per chunk)
            Example: [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
        
        Note: Chroma.from_documents() calls this internally
              to embed all chunks before storing them.
        """
        return self.embeddings.embed_documents(texts)