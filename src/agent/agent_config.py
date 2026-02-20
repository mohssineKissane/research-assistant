"""
Agent Configuration - Controlling Agent Behavior
=================================================

This module defines configuration for the research agent, including:
- Agent behavior settings (iterations, verbosity)
- Tool-specific parameters
- LLM settings for agent reasoning
- Custom prompts to guide agent decision-making

Prompt Engineering for Agents:
    The prefix and suffix prompts are crucial for agent performance.
    They tell the agent:
    - What its role is (research assistant)
    - What tools it has access to
    - How to approach problems (search docs first, then web)
    - How to format answers (cite sources)
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class AgentConfig:
    """
    Configuration for research agent behavior and capabilities.
    
    This dataclass centralizes all agent settings, making it easy to:
    - Adjust agent behavior without changing code
    - Create different agent profiles (conservative vs exploratory)
    - Tune performance based on use case
    """
    
    # ==================== Agent Behavior ====================
    
    agent_type: str = "zero-shot-react-description"
    """Type of agent reasoning pattern (zero-shot-react-description recommended)"""
    
    verbose: bool = True
    """If True, shows agent's thought process (Thought/Action/Observation loop)"""
    
    max_iterations: int = 5
    """Maximum tool calls before stopping (prevents infinite loops)"""
    
    # ==================== Tool Settings ====================
    
    document_search_k: int = 4
    """Number of document chunks to retrieve per search (more = more context but slower)"""
    
    web_search_max_results: int = 5
    """Maximum web search results to return"""
    
    # ==================== LLM Settings ====================
    
    temperature: float = 0.7
    """
    Controls randomness in agent reasoning:
    - 0.0 = Deterministic, focused (good for factual tasks)
    - 0.7 = Balanced creativity and consistency (recommended)
    - 1.0 = More creative but less predictable
    """
    
    max_tokens: int = 2000
    """Maximum tokens in agent's response"""
    
    # ==================== Prompt Customization ====================
    
    agent_prefix: Optional[str] = None
    """
    Custom instruction prefix for the agent.
    This sets the agent's role and guidelines.
    If None, uses LangChain's default prefix.
    """
    
    agent_suffix: Optional[str] = None
    """
    Custom instruction suffix for the agent.
    This provides final reminders before the agent starts reasoning.
    If None, uses LangChain's default suffix.
    """
    
    def get_agent_kwargs(self):
        """
        Build kwargs dict for agent initialization.
        
        This method converts the config into the format expected by
        LangChain's initialize_agent() function.
        
        It automatically handles default prompts:
        - If custom prefix/suffix is set, uses that
        - If not, uses the default research agent prompts
        
        Returns:
            Dict of agent initialization parameters
        """
        kwargs = {
            'verbose': self.verbose,
            'max_iterations': self.max_iterations,
            'handle_parsing_errors': True,  # Gracefully handle malformed LLM outputs
            'early_stopping_method': 'generate'  # Stop when confident answer is found
        }
        
        # Prepare agent prompts (use custom if set, else defaults)
        agent_kwargs = {
            'prefix': self.agent_prefix or self.get_research_agent_prefix(),
            'suffix': self.agent_suffix or self.get_research_agent_suffix()
        }
            
        kwargs['agent_kwargs'] = agent_kwargs
        
        return kwargs  
    
    @staticmethod
    def get_research_agent_prefix():
        """
        Custom prefix that defines the agent's role and guidelines.
        
        This prompt engineering is critical for agent performance:
        - Defines the agent's identity (research assistant)
        - Lists the agent's capabilities (search, summarize, combine)
        - Sets expectations (cite sources, use tools)
        
        The prefix appears BEFORE the tool descriptions in the agent's prompt.
        
        Returns:
            String containing the agent's role and instructions
        """
        return """You are a research assistant with access to tools.

        Your goal is to help users research topics by:
        1. Searching uploaded documents for relevant information
        2. Searching the web when documents don't have the answer
        3. Summarizing information when requested
        4. Combining multiple sources for comprehensive answers

        Always cite your sources and be clear about where information comes from.
        If you're not sure, use the tools to find out.

        You have access to the following tools:"""
    
    @staticmethod
    def get_research_agent_suffix():
        """
        Custom suffix with final reminders before agent starts reasoning.
        
        This appears AFTER the tool descriptions and provides:
        - Final behavioral guidelines
        - Reminders about best practices
        - The template for the ReAct loop (Question → Thought → Action)
        
        The {input} and {agent_scratchpad} are placeholders filled by LangChain:
        - {input}: The user's question
        - {agent_scratchpad}: The agent's reasoning history (Thought/Action/Observation)
        
        Returns:
            String containing final instructions and prompt template
        """
        return """Begin! Remember to:
        - Use tools to find information
        - Cite sources in your final answer
        - Be concise but comprehensive
        - If documents don't have info, try web search

Question: {input}
Thought: {agent_scratchpad}"""

    @staticmethod
    def get_conversational_agent_prefix():
        """
        Strong prefix for conversational-react-description agent.
        
        This agent type requires the prefix only (the suffix with {chat_history},
        {input}, {agent_scratchpad} is handled by LangChain's built-in template).
        
        The key difference from zero-shot: ALWAYS use tools first, answer later.
        """
        return """You are a research assistant that ALWAYS uses tools to find information before answering.

You have access to these tools:
- search_documents: Search the uploaded PDF documents
- search_web: Search the internet for current or missing information
- summarize_content: Summarize document content

CRITICAL RULES:
1. ALWAYS use at least one tool before giving a final answer
2. For questions about recent events, current AI models, news, or anything after 2023: use search_web
3. For questions about uploaded documents: use search_documents
4. For summary requests: use summarize_content
5. You may call multiple tools in sequence
6. Never say you don't have access to information without trying search_web first

You have access to the following tools:"""

    @staticmethod
    def get_conversational_agent_suffix():
        """
        Suffix for conversational-react-description.
        MUST include {chat_history}, {input}, and {agent_scratchpad}.
        """
        return """Begin! You MUST use a tool before providing a Final Answer.

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}"""