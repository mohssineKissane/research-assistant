"""
LLM Manager for Groq Integration
Provides easy access to Groq LLM with configuration from config.yaml
"""
from langchain_groq import ChatGroq
from src.utils.config import config


class LLMManager:
    """Manager for Groq LLM instances"""
    
    def __init__(self):
        """Initialize LLM Manager with config"""
        self.llm = None
    
    def get_llm(self, temperature=None, model_name=None, max_tokens=None):
        """
        Get Groq LLM instance
        
        Args:
            temperature: 0 = deterministic, 1 = creative (defaults to config)
            model_name: Model to use (defaults to config)
            max_tokens: Maximum tokens in response (defaults to config)
            
        Returns:
            ChatGroq instance
            
        Example:
            from src.utils.llm import llm_manager
            
            # Use defaults from config.yaml
            llm = llm_manager.get_llm()
            
            # Or override specific values
            llm = llm_manager.get_llm(temperature=0.9, model_name="llama-3.1-8b-instant")
        """
        # Use config values as defaults
        if temperature is None:
            temperature = config.llm_temperature
        if model_name is None:
            model_name = config.llm_model_name
        if max_tokens is None:
            max_tokens = config.llm_max_tokens
        
        # Create ChatGroq instance with parameters compatible with langchain-groq==0.1.3
        self.llm = ChatGroq(
            groq_api_key=config.groq_api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return self.llm


# Create global LLM manager instance
llm_manager = LLMManager()