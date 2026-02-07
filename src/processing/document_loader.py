"""
Document Loader - Loading PDFs into LangChain Documents
========================================================

This module handles loading PDF files and converting them to LangChain
Document objects - the standard format for text + metadata in LangChain.

What is a LangChain Document?
    A simple container with two parts:
    - page_content: The text content (string)
    - metadata: Dictionary of information about the text
    
    Example:
        Document(
            page_content="Artificial Intelligence is...",
            metadata={
                'source': 'paper.pdf',
                'page': 0,
                'filename': 'paper.pdf',
                'upload_date': '2024-01-15T10:30:00'
            }
        )

Why Metadata Matters:
    - Citations: Know which file and page an answer came from
    - Filtering: Search only certain documents
    - Debugging: Trace where information originated
"""
from typing import List
from datetime import datetime
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document


class DocumentLoader:
    """
    Handles loading PDF files into LangChain Document format.
    
    Each page of a PDF becomes a separate Document object.
    Custom metadata is added for tracking and citations.
    
    LangChain has many loaders for different file types:
    - PyPDFLoader: PDF files
    - TextLoader: Plain text files
    - CSVLoader: CSV files
    - UnstructuredWordDocumentLoader: Word docs
    - WebBaseLoader: Web pages
    
    They all produce the same output: List[Document]
    """
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load a single PDF file.
        
        How PyPDFLoader works:
            1. Opens PDF with PyPDF library
            2. Extracts text from each page
            3. Creates one Document per page
            4. Adds basic metadata (source, page number)
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects, one per page.
            
        Example output for a 3-page PDF:
            [
                Document(page_content="Page 1 text...", metadata={'source': 'file.pdf', 'page': 0}),
                Document(page_content="Page 2 text...", metadata={'source': 'file.pdf', 'page': 1}),
                Document(page_content="Page 3 text...", metadata={'source': 'file.pdf', 'page': 2})
            ]
        """
        # PyPDFLoader is LangChain's wrapper around pypdf
        loader = PyPDFLoader(file_path)
        
        # load() reads the PDF and returns List[Document]
        documents = loader.load()
        
        # Add custom metadata for better tracking
        # This metadata travels with the document through the entire pipeline
        for doc in documents:
            # Add filename (just the file name, not full path)
            doc.metadata['filename'] = os.path.basename(file_path)
            
            # Add upload timestamp
            # Note: Using ISO format string because ChromaDB requires serializable values
            doc.metadata['upload_date'] = datetime.now().isoformat()
        
        return documents
    
    def load_multiple_pdfs(self, file_paths: List[str]) -> List[Document]:
        """
        Load multiple PDF files.
        
        All documents are combined into a single list.
        Each document retains its source metadata.
        
        Args:
            file_paths: List of paths to PDF files
            
        Returns:
            Combined list of all Document objects from all files.
            
        Example:
            docs = loader.load_multiple_pdfs([
                "paper1.pdf",  # 10 pages
                "paper2.pdf"   # 5 pages
            ])
            len(docs)  # → 15 Documents
        """
        all_docs = []
        for path in file_paths:
            all_docs.extend(self.load_pdf(path))
        return all_docs