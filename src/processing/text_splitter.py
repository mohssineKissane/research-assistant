from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class DocumentSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        """
        chunk_size: max characters per chunk
        chunk_overlap: overlap between chunks (preserves context)
        """
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