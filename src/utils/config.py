"""
Configuration Manager
Loads configuration from config.yaml and environment variables
"""
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv


class Config:
    """Singleton configuration manager"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Load environment variables
        load_dotenv()
        
        # Load YAML configuration
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        self._initialized = True
    
    # Environment variables
    @property
    def groq_api_key(self):
        """Get Groq API key from environment"""
        return os.getenv("GROQ_API_KEY")
    
    # LLM Configuration
    def get_llm_config(self):
        """Get LLM configuration"""
        return self._config.get('llm', {})
    
    @property
    def llm_model_name(self):
        """Get LLM model name"""
        return self._config.get('llm', {}).get('model_name', 'llama-3.1-70b-versatile')
    
    @property
    def llm_temperature(self):
        """Get LLM temperature"""
        return self._config.get('llm', {}).get('temperature', 0.7)
    
    @property
    def llm_max_tokens(self):
        """Get LLM max tokens"""
        return self._config.get('llm', {}).get('max_tokens', 2048)
    
    # Embeddings Configuration
    def get_embeddings_config(self):
        """Get embeddings configuration"""
        return self._config.get('embeddings', {})
    
    @property
    def embeddings_model_name(self):
        """Get embeddings model name"""
        return self._config.get('embeddings', {}).get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
    
    @property
    def embeddings_device(self):
        """Get embeddings device (cpu/cuda)"""
        return self._config.get('embeddings', {}).get('device', 'cpu')
    
    @property
    def embeddings_normalize(self):
        """Get embeddings normalize flag"""
        return self._config.get('embeddings', {}).get('normalize', True)
    
    # Vector Store Configuration
    def get_vectorstore_config(self):
        """Get vector store configuration"""
        return self._config.get('vectorstore', {})
    
    @property
    def chunk_size(self):
        """Get chunk size for text splitting"""
        return self._config.get('vectorstore', {}).get('chunk_size', 1000)
    
    @property
    def chunk_overlap(self):
        """Get chunk overlap for text splitting"""
        return self._config.get('vectorstore', {}).get('chunk_overlap', 200)
    
    @property
    def vectorstore_persist_directory(self):
        """Get vector store persist directory"""
        return self._config.get('vectorstore', {}).get('persist_directory', './data/vectorstore')


# Create global config instance
config = Config()