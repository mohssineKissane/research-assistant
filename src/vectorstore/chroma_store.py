from typing import List
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

class ChromaVectorStore:
    def __init__(self, embeddings, persist_directory="./data/vectorstore"):
        """
        embeddings: EmbeddingsGenerator instance
        persist_directory: where to save the database
        """
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.vectorstore = None
    
    def create_from_documents(self, documents: List[Document], collection_name="research_docs"):
        """Create new vectorstore from documents"""
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings.get_embeddings(),
            persist_directory=self.persist_directory,
            collection_name=collection_name
        )
        return self.vectorstore
    
    def load_existing(self, collection_name="research_docs"):
        """Load existing vectorstore"""
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings.get_embeddings(),
            collection_name=collection_name
        )
        return self.vectorstore
    
    def add_documents(self, documents: List[Document]):
        """Add more documents to existing store"""
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized")
        self.vectorstore.add_documents(documents)
    
    def similarity_search(self, query: str, k=4):
        """Find k most similar chunks to query"""
        return self.vectorstore.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k=4):
        """Search with relevance scores"""
        return self.vectorstore.similarity_search_with_score(query, k=k)