"""
Web Search Tool - Internet Search Using Tavily API
===================================================

This tool enables the agent to search the internet using Tavily Search API,
which is specifically designed for AI agents and LLMs.

Tavily Benefits:
    - Designed specifically for AI/LLM use cases
    - Returns clean, relevant results optimized for agents
    - No complex setup (just API key)
    - Fast and reliable
    - 1000 free searches per month

Agent Decision-Making:
    The agent chooses between document_search and web_search based on:
    - Document search: For information IN the uploaded PDFs
    - Web search: For current events, recent developments, or missing info
    
Example Agent Reasoning:
    User: "What does the paper say about GPT-4, and what's new in GPT-5?"
    
    Thought: First part needs document search, second needs web
    Action 1: search_documents("GPT-4")
    Observation 1: [Found GPT-4 info in paper]
    
    Thought: Now need current info about GPT-5
    Action 2: search_web("GPT-5 latest developments")
    Observation 2: [Found recent articles]
    
    Final Answer: [Combines both sources]
"""
from langchain.tools import BaseTool
from pydantic import Field
from typing import Any
import os

class WebSearchTool(BaseTool):
    """
    Tool for searching the internet using Tavily Search API.
    
    Tavily is specifically designed for AI agents and provides:
    - Clean, relevant results optimized for LLMs
    - Current events and news
    - Information not in uploaded documents
    - Fast and reliable search
    """
    
    name: str = "search_web"
    """Tool identifier for agent to call"""
    
    description: str = """
    Use this tool to search the internet for current information.
    
    When to use:
    - User asks about recent events or news
    - Need information not in uploaded documents
    - Looking for current data, statistics, or developments
    - Questions about events after document publication dates
    
    Input: A search query
    Output: Recent web search results with sources
    """
    """
    The description tells the agent this is for CURRENT/EXTERNAL information,
    while document_search is for information IN the uploaded PDFs.
    """
    
    max_results: int = 5
    
    class Config:
        """Pydantic config to exclude max_results from tool schema"""
        arbitrary_types_allowed = True

    
    def _run(self, query: str) -> str:
        """
        Perform web search using Tavily Search API.
        
        Tavily is used because:
        - Specifically designed for AI agents
        - Returns clean, relevant results
        - Simple setup (just API key)
        - Fast and reliable
        - 1000 free searches per month
        
        Args:
            query: Search query from the agent
            
        Returns:
            Formatted string with search results, URLs, and snippets
        """
        try:
            # Import Tavily client
            from tavily import TavilyClient
            from dotenv import load_dotenv
            
            # Ensure .env file is loaded (in case tool is used standalone)
            load_dotenv()
            
            # Get API key from environment variables
            api_key = os.getenv('TAVILY_API_KEY')
            
            # Validate credentials
            if not api_key or api_key == 'your_tavily_api_key_here':
                return (
                    "Tavily API key not configured. "
                    "Please set TAVILY_API_KEY in your .env file. "
                    "Get your free API key at: https://tavily.com"
                )
            
            # Initialize Tavily client
            client = TavilyClient(api_key=api_key)
            
            # Perform search
            # search_depth="basic" for faster results, "advanced" for more thorough
            response = client.search(
                query=query,
                max_results=self.max_results,
                search_depth="basic",
                include_answer=False  # We'll format our own answer
            )
            
            # Check if we got results
            if not response or 'results' not in response or not response['results']:
                return "No web results found for this query."
            
            # Format results for the agent
            output = "Found the following information from the web:\n\n"
            
            for i, result in enumerate(response['results'], 1):
                title = result.get('title', 'No title')
                content = result.get('content', 'No description')
                url = result.get('url', '')
                
                # Provide structured output so agent can cite sources
                output += f"[{i}] {title}\n"
                output += f"URL: {url}\n"
                output += f"Summary: {content}\n\n"
            
            return output
            
        except ImportError:
            return (
                "Tavily library not installed. "
                "Run: uv pip install tavily-python"
            )
        
        except Exception as e:
            error_msg = str(e)
            
            # Handle specific error cases with helpful messages
            if "api key" in error_msg.lower() or "unauthorized" in error_msg.lower():
                return (
                    "Invalid Tavily API key. "
                    "Please check your TAVILY_API_KEY in .env file. "
                    "Get a free key at: https://tavily.com"
                )
            elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
                return (
                    "Tavily API quota exceeded. "
                    "Free tier allows 1000 searches per month. "
                    "Upgrade at: https://tavily.com/pricing"
                )
            else:
                # Return error to agent (agent might try rephrasing or use another tool)
                return f"Error performing web search: {error_msg}"
    
    async def _arun(self, query: str) -> str:
        """
        Async version for concurrent execution.
        
        Currently calls sync version.
        Future: Could use async HTTP client for better performance.
        """
        return self._run(query)

