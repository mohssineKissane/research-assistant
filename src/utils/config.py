"""
Configuration Manager - Centralized Settings
=============================================

This module provides centralized configuration management.
Loads settings from config.yaml and environment variables (.env).

Configuration Sources:
    1. config.yaml: Application settings (chunk size, model names, etc.)
    2. .env file: Secrets (API keys)
    
Why Centralized Config?
    - Change settings without modifying code
    - Keep secrets out of version control
    - Easy to switch between environments (dev/prod)
    
Singleton Pattern:
    Only one Config instance exists in the entire application.
    All modules import the same 'config' object.
    This ensures consistent settings everywhere.

Settings in config.yaml:
    llm:
      model_name: "llama-3.3-70b-versatile"
      temperature: 0.7
      max_tokens: 2048
    
    embeddings:
      model_name: "sentence-transformers/all-MiniLM-L6-v2"
      device: "cpu"
      normalize: true
    
    vectorstore:
      chunk_size: 1000
      chunk_overlap: 200
      persist_directory: "./data/vectorstore"
"""
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv


class Config:
    """
    Singleton configuration manager.
    
    Loads settings from:
    - config.yaml (application settings)
    - .env file (secrets like API keys)
    
    Provides properties to access settings throughout the application.
    
    Singleton Pattern:
        Config() always returns the same instance.
        First call initializes, subsequent calls reuse.
    """
    _instance = None  # Class-level storage for singleton instance
    
    def __new__(cls):
        """
        Implement singleton pattern.
        
        If no instance exists, create one.
        If instance exists, return the existing one.
        """
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """
        Initialize configuration (only runs once due to singleton).
        
        Steps:
        1. Load .env file (makes env vars available)
        2. Load config.yaml (parses YAML into dict)
        3. Mark as initialized
        """
        # Skip if already initialized (singleton)
        if self._initialized:
            return
        
        # Load environment variables from .env file
        # This makes GROQ_API_KEY available via os.getenv()
        load_dotenv()
        
        # Find and load config.yaml
        # Path is relative to this file: ../../config.yaml
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Parse YAML into dictionary
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        self._initialized = True
    
    # ==================== Environment Variables ====================
    
    @property
    def groq_api_key(self):
        """
        Get Groq API key from environment.
        
        Set in .env file:
            GROQ_API_KEY=gsk_your_key_here
        
        Never commit this to version control!
        """
        return os.getenv("GROQ_API_KEY")
    
    # ==================== LLM Configuration ====================
    
    def get_llm_config(self):
        """Get entire LLM configuration section as dict."""
        return self._config.get('llm', {})
    
    @property
    def llm_model_name(self):
        """
        Get LLM model name.
        
        Options:
        - llama-3.3-70b-versatile (best quality)
        - llama-3.1-8b-instant (faster)
        """
        return self._config.get('llm', {}).get('model_name', 'llama-3.1-70b-versatile')
    
    @property
    def llm_temperature(self):
        """
        Get LLM temperature.
        
        Controls randomness:
        - 0.0: Deterministic (same input → same output)
        - 0.7: Balanced (good for RAG)
        - 1.0: Very creative (may hallucinate)
        """
        return self._config.get('llm', {}).get('temperature', 0.7)
    
    @property
    def llm_max_tokens(self):
        """
        Get max tokens for LLM response.
        
        Limits response length:
        - 1024: Short responses
        - 2048: Medium (default)
        - 4096: Long responses
        """
        return self._config.get('llm', {}).get('max_tokens', 2048)
    
    # ==================== Embeddings Configuration ====================
    
    def get_embeddings_config(self):
        """Get entire embeddings configuration section as dict."""
        return self._config.get('embeddings', {})
    
    @property
    def embeddings_model_name(self):
        """
        Get embeddings model name.
        
        Default: sentence-transformers/all-MiniLM-L6-v2
        - 384 dimensions
        - Fast, runs locally
        - Good quality for general use
        """
        return self._config.get('embeddings', {}).get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
    
    @property
    def embeddings_device(self):
        """
        Get device for embeddings model.
        
        Options:
        - "cpu": Works everywhere (default)
        - "cuda": Use GPU if available (faster)
        """
        return self._config.get('embeddings', {}).get('device', 'cpu')
    
    @property
    def embeddings_normalize(self):
        """
        Whether to normalize embeddings to unit length.
        
        Should be True for cosine similarity search (default).
        Normalization: vector / ||vector||
        """
        return self._config.get('embeddings', {}).get('normalize', True)
    
    # ==================== Vector Store Configuration ====================
    
    def get_vectorstore_config(self):
        """Get entire vector store configuration section as dict."""
        return self._config.get('vectorstore', {})
    
    @property
    def chunk_size(self):
        """
        Get target chunk size for text splitting.
        
        Measured in characters.
        - 500: Small chunks, precise retrieval
        - 1000: Balanced (default)
        - 2000: Larger chunks, more context per chunk
        """
        return self._config.get('vectorstore', {}).get('chunk_size', 1000)
    
    @property
    def chunk_overlap(self):
        """
        Get chunk overlap for text splitting.
        
        Characters shared between adjacent chunks.
        Preserves context at chunk boundaries.
        Typically 10-20% of chunk_size.
        """
        return self._config.get('vectorstore', {}).get('chunk_overlap', 200)
    
    @property
    def vectorstore_persist_directory(self):
        """
        Get directory for persisting ChromaDB data.
        
        ChromaDB saves data here so you can reload
        without re-indexing documents.
        """
        return self._config.get('vectorstore', {}).get('persist_directory', './data/vectorstore')


# Create global config instance (singleton)
# Import this anywhere: from src.utils.config import config
config = Config()