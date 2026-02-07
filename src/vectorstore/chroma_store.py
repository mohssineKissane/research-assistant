"""
ChromaDB Vector Store - The Semantic Search Database
=====================================================

This module manages the ChromaDB vector database for storing and searching
document embeddings.

What is a Vector Store?
    A database optimized for storing and searching high-dimensional vectors.
    Instead of keyword matching, it finds documents with similar MEANING.

How ChromaDB Works:
    1. Store: Document chunk + its embedding vector + metadata
    2. Search: Query vector → Find nearest vectors → Return documents
    
    Behind the scenes:
    - Vectors are indexed using approximate nearest neighbor (ANN) algorithms
    - Search is O(log n) not O(n) - scales to millions of documents
    - Uses cosine similarity: dot(v1, v2) / (||v1|| * ||v2||)

Persistence:
    ChromaDB saves data to disk (persist_directory).
    You can close the app and reload existing vectors later.

Key Insight:
    When you create a ChromaDB store, you give it an embeddings object.
    ChromaDB keeps a reference to it. During search, it uses this
    same embeddings object to convert the query to a vector.
"""
from typing import List
from langchain_community.vectorstores import Chroma
from langchain.schema import Document


class ChromaVectorStore:
    """
    Wrapper around LangChain's Chroma vectorstore.
    
    This class provides a clean interface for:
    - Creating a new vector store from documents
    - Loading an existing vector store
    - Adding more documents
    - Searching for similar chunks
    
    The key responsibility is managing the connection between
    documents, embeddings, and the underlying ChromaDB database.
    """
    
    def __init__(self, embeddings, persist_directory="./data/vectorstore"):
        """
        Initialize the ChromaDB vector store.
        
        Args:
            embeddings: EmbeddingsGenerator instance
                       This is STORED and used later for:
                       - Embedding documents during indexing
                       - Embedding queries during search
                       
                       CRITICAL: This same object must be used for both!
                       
            persist_directory: Where to save ChromaDB files on disk.
                              Data persists across program restarts.
        
        After init:
            self.embeddings = your EmbeddingsGenerator
            self.vectorstore = None (until you create or load one)
        """
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.vectorstore = None  # Will be set by create_from_documents or load_existing
    
    def create_from_documents(self, documents: List[Document], collection_name="research_docs"):
        """
        Create a new vector store from documents.
        
        This is the INDEXING step - converting documents to searchable vectors.
        
        What happens inside Chroma.from_documents():
            for each doc in documents:
                1. embedding = self.embeddings.embed_documents([doc.page_content])
                   → [0.15, -0.32, 0.78, ...] (384 numbers)
                2. Store in ChromaDB:
                   - ID: auto-generated
                   - Vector: the embedding
                   - Document: page_content
                   - Metadata: {filename, page, chunk_id, upload_date}
        
        Args:
            documents: List of LangChain Document objects
                      Each Document has:
                      - page_content: The text chunk
                      - metadata: Dict with source info
                      
            collection_name: Name for this set of documents in ChromaDB.
                            Allows multiple collections in same database.
        
        Returns:
            Chroma vectorstore instance (also stored in self.vectorstore)
        """
        # Chroma.from_documents does:
        # 1. Embeds all documents using self.embeddings.get_embeddings()
        # 2. Stores vectors + documents in ChromaDB
        # 3. Saves to persist_directory
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            
            # This embeddings object is stored by Chroma as _embedding_function
            # It will be used later to embed search queries
            embedding=self.embeddings.get_embeddings(),
            
            persist_directory=self.persist_directory,
            collection_name=collection_name
        )
        return self.vectorstore
    
    def load_existing(self, collection_name="research_docs"):
        """
        Load an existing vector store from disk.
        
        Use this when you've already indexed documents and want to
        reload them without re-processing.
        
        Args:
            collection_name: Name of the collection to load
            
        Returns:
            Chroma vectorstore instance
            
        Note: You MUST provide the same embeddings model that was
              used during indexing, otherwise search won't work correctly.
        """
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            
            # Same embeddings model - required for correct search
            embedding_function=self.embeddings.get_embeddings(),
            
            collection_name=collection_name
        )
        return self.vectorstore
    
    def add_documents(self, documents: List[Document]):
        """
        Add more documents to an existing vector store.
        
        Useful for incremental updates without reprocessing everything.
        
        Args:
            documents: New documents to add
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized")
        self.vectorstore.add_documents(documents)
    
    def similarity_search(self, query: str, k=4):
        """
        Find k most similar document chunks to the query.
        
        This is the SEARCH step - finding relevant context for a question.
        
        What happens:
            1. query → self._embedding_function.embed_query(query)
               "What is AI?" → [0.12, -0.45, ...]
               
            2. ChromaDB computes cosine similarity with ALL stored vectors
               similarity = dot(query_vec, doc_vec) / (||q|| * ||d||)
               
            3. Returns k documents with highest similarity scores
        
        Args:
            query: The search query (e.g., user's question)
            k: Number of results to return (default: 4)
            
        Returns:
            List of k Document objects most similar to the query.
            Each Document has page_content and metadata.
        """
        return self.vectorstore.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k=4):
        """
        Search with relevance scores (useful for debugging).
        
        Returns:
            List of (Document, score) tuples.
            Score is cosine similarity (0 to 1, higher = more similar).
        """
        return self.vectorstore.similarity_search_with_score(query, k=k)