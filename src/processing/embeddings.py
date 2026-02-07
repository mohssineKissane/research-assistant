from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.utils.config import config

class EmbeddingsGenerator:
    def __init__(self, model_name=None, device=None, normalize=None):
        """
        Use HuggingFace free model
        Runs locally, no API calls
        
        Args:
            model_name: Model name (defaults to config)
            device: Device to use - 'cpu' or 'cuda' (defaults to config)
            normalize: Whether to normalize embeddings (defaults to config)
        """
        # Use config values as defaults
        if model_name is None:
            model_name = config.embeddings_model_name
        if device is None:
            device = config.embeddings_device
        if normalize is None:
            normalize = config.embeddings_normalize
            
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': normalize}
        )
    
    def get_embeddings(self):
        """Return embeddings instance for vectorstore"""
        return self.embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.embeddings.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        return self.embeddings.embed_documents(texts)