"""
LLM Manager - Groq API Integration
====================================

This module manages the connection to Groq's LLM API.
Groq provides fast inference for open-source models like Llama.

Why Groq?
    - Fast inference (specialized hardware)
    - Free tier available for development
    - Access to latest Llama models
    - LangChain integration via langchain-groq

LLM (Large Language Model):
    The AI model that generates text responses.
    In RAG, the LLM receives context + question and generates the answer.
    
    Parameters:
    - temperature: 0 = deterministic, 1 = creative
    - max_tokens: Maximum tokens in response
    - model_name: Which model to use
    
Current Model: llama-3.3-70b-versatile
    - 70 billion parameters
    - Good at following instructions
    - Supports long context windows
"""
from langchain_groq import ChatGroq
from src.utils.config import config


class LLMManager:
    """
    Manager for Groq LLM instances.
    
    Provides a clean interface to get LLM instances with:
    - Default settings from config.yaml
    - Optional overrides per call
    - Singleton pattern (reuse same instance)
    
    The LLM is used in the RAG pipeline for:
    - Generating answers based on retrieved context
    - Following prompt instructions (cite sources, stay grounded)
    """
    
    def __init__(self):
        """
        Initialize the LLM Manager.
        
        Does NOT create an LLM instance yet - that happens in get_llm().
        This allows lazy initialization and parameter customization.
        """
        self.llm = None  # Created lazily by get_llm()
    
    def get_llm(self, temperature=None, model_name=None, max_tokens=None):
        """
        Get a Groq LLM instance.
        
        Creates a ChatGroq instance with specified or default parameters.
        
        Args:
            temperature: Controls randomness (0.0 to 1.0)
                        - 0.0: Deterministic (same input → same output)
                        - 0.3: Slightly creative
                        - 0.7: Balanced (default for RAG)
                        - 1.0: Very creative (may hallucinate more)
                        
            model_name: Groq model identifier
                       Current options:
                       - "llama-3.3-70b-versatile" (best quality)
                       - "llama-3.1-8b-instant" (faster, smaller)
                       - "mixtral-8x7b-32768" (good alternative)
                       
            max_tokens: Maximum tokens in response
                       - 2048: Good for detailed answers
                       - 1024: Shorter, faster responses
                       - 4096: Very long responses (if needed)
        
        Returns:
            ChatGroq instance ready to use
            
        Example usage:
            # Use defaults from config.yaml
            llm = llm_manager.get_llm()
            
            # Override for a specific use case
            llm = llm_manager.get_llm(temperature=0.9)
            
            # Direct invocation
            response = llm.invoke("Hello!")
            print(response.content)
        """
        # Use config values as defaults
        # Allows runtime overrides without changing config.yaml
        if temperature is None:
            temperature = config.llm_temperature
        if model_name is None:
            model_name = config.llm_model_name
        if max_tokens is None:
            max_tokens = config.llm_max_tokens
        
        # Create ChatGroq instance
        # ChatGroq is LangChain's wrapper around Groq's API
        self.llm = ChatGroq(
            # API key from .env file (via config)
            groq_api_key=config.groq_api_key,
            
            # Model to use
            model_name=model_name,
            
            # Generation parameters
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return self.llm


# Create global LLM manager instance (singleton pattern)
# Import this and use llm_manager.get_llm() anywhere in the project
llm_manager = LLMManager()