from typing import List
from datetime import datetime
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

class DocumentLoader:
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load PDF and return Document objects
        Each page becomes a Document with metadata
        """
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Add custom metadata
        for doc in documents:
            doc.metadata['filename'] = os.path.basename(file_path)
            doc.metadata['upload_date'] = datetime.now()
        
        return documents
    
    def load_multiple_pdfs(self, file_paths: List[str]) -> List[Document]:
        """Load multiple PDFs"""
        all_docs = []
        for path in file_paths:
            all_docs.extend(self.load_pdf(path))
        return all_docs