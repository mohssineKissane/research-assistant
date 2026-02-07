"""
Text Splitter - Chunking Documents for Effective Retrieval
===========================================================

This module splits large documents into smaller, overlapping chunks.
Chunking is crucial for RAG - it determines what gets retrieved.

Why Chunk Documents?
    1. LLMs have context limits (can't process entire documents at once)
    2. Smaller chunks = more precise retrieval (find the RIGHT part)
    3. Embedding quality decreases with very long text
    
Chunk Size Trade-offs:
    - Too small (100 chars): Loses context, fragments sentences
    - Too big (5000 chars): Imprecise retrieval, may exceed LLM context
    - Sweet spot (500-1500 chars): Good balance for most use cases
    
Why Overlap?
    If a key sentence is split between chunks, overlap ensures it appears
    in at least one chunk in full. Preserves context at boundaries.
    
    Without overlap: [chunk1][chunk2] - sentence might be split
    With overlap:    [chunk1----|----chunk2] - key info preserved

RecursiveCharacterTextSplitter:
    LangChain's smart splitter that tries to keep semantic units together.
    Splits in order: paragraphs → sentences → words → characters
"""
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from src.utils.config import config


class DocumentSplitter:
    """
    Splits documents into overlapping chunks for better retrieval.
    
    Uses LangChain's RecursiveCharacterTextSplitter which:
    1. First tries to split on paragraph breaks (\\n\\n)
    2. If chunks still too big, splits on newlines (\\n)
    3. Then sentences (". ")
    4. Then spaces (" ")
    5. Finally characters if needed
    
    This recursive approach preserves semantic units when possible.
    """
    
    def __init__(self, chunk_size=None, chunk_overlap=None):
        """
        Initialize the document splitter.
        
        Args:
            chunk_size: Maximum characters per chunk (default: 1000 from config)
                       This is the target size, actual chunks may vary slightly.
                       
            chunk_overlap: Characters to overlap between chunks (default: 200 from config)
                          Typically 10-20% of chunk_size.
                          
        Example with chunk_size=1000, chunk_overlap=200:
            Original text (2500 chars):
            [==========================================]
            
            Chunks created:
            [chunk 1: chars 0-1000    ]
                    [chunk 2: chars 800-1800   ]
                            [chunk 3: chars 1600-2500]
            
            Each chunk shares 200 chars with its neighbor.
        """
        # Use config values as defaults
        if chunk_size is None:
            chunk_size = config.chunk_size
        if chunk_overlap is None:
            chunk_overlap = config.chunk_overlap
        
        # Store for reference
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Create the LangChain splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            
            # Separators to try, in order of preference
            # Splitter tries to split on these before falling back to smaller units
            separators=[
                "\n\n",   # Paragraph breaks (most preferred - keeps paragraphs intact)
                "\n",     # Line breaks
                ". ",     # Sentence endings
                " ",      # Word boundaries
                ""        # Individual characters (last resort)
            ],
            
            # How to measure chunk size (default: character count)
            length_function=len,
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks, preserving metadata.
        
        For each input document:
        1. Text is split into chunks based on separators
        2. Each chunk becomes a new Document
        3. Original metadata is copied to each chunk
        4. chunk_id is added to track position
        
        Args:
            documents: List of Document objects (e.g., from DocumentLoader)
            
        Returns:
            List of smaller Document objects (more documents, less text each)
            
        Example:
            Input:  2 Documents (one per page), each 5000 chars
            Output: ~10 Documents (chunks), each ~1000 chars
            
        Metadata preserved:
            Original: {source: 'file.pdf', page: 0, filename: 'file.pdf'}
            Chunk:    {source: 'file.pdf', page: 0, filename: 'file.pdf', chunk_id: 3}
        """
        # split_documents preserves metadata from original documents
        chunks = self.splitter.split_documents(documents)
        
        # Add chunk index to metadata for tracking
        # This helps identify which chunk a result came from
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
        
        return chunks