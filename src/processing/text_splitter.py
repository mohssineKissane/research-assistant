from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from src.utils.config import config

class DocumentSplitter:
    def __init__(self, chunk_size=None, chunk_overlap=None):
        """
        chunk_size: max characters per chunk (defaults to config)
        chunk_overlap: overlap between chunks (defaults to config)
        """
        # Use config values as defaults
        if chunk_size is None:
            chunk_size = config.chunk_size
        if chunk_overlap is None:
            chunk_overlap = config.chunk_overlap
            
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],  # Try these in order
            length_function=len,
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split docs into chunks, preserving metadata"""
        chunks = self.splitter.split_documents(documents)
        
        # Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
        
        return chunks