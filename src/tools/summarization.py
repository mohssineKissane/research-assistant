"""
Summarization Tool - Condensing Documents and Content
======================================================

This tool enables the agent to create summaries of documents or search results.

When the Agent Uses This Tool:
    The agent recognizes summarization requests through keywords:
    - "summarize", "summary", "overview"
    - "brief me", "give me the gist"
    - "what's the main idea"
    
LangChain's Map-Reduce Pattern:
    For multi-document summarization, this tool uses map-reduce:
    
    MAP phase:
        Doc 1 → Summarize → Summary 1
        Doc 2 → Summarize → Summary 2
        Doc 3 → Summarize → Summary 3
    
    REDUCE phase:
        Summary 1 + Summary 2 + Summary 3 → Combine → Final Summary
    
    This allows summarizing content longer than the LLM's context window.
"""
from langchain.tools import BaseTool
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from pydantic import Field
from typing import Any

class SummarizationTool(BaseTool):
    """
    Tool for summarizing documents or content.
    
    This tool provides two modes:
    1. Full document summary: Summarizes all uploaded documents
    2. Targeted summary: Summarizes content related to a specific query
    """
    
    name: str = "summarize_content"
    """Tool identifier for agent"""
    
    description: str = """
    Use this tool to create summaries of documents or long content.
    
    When to use:
    - User asks for a summary or overview
    - Need to condense long information
    - Asked to "summarize", "give overview", or "brief me"
    
    Input: Content to summarize (or "all documents" for full summary)
    Output: Concise summary of the content
    """
    """
    The description helps the agent recognize summarization requests.
    Keywords like "summarize", "overview", "brief" trigger this tool.
    """
    
    llm: Any = Field(exclude=True)
    """LLM for generating summaries"""
    
    vectorstore: Any = Field(exclude=True)
    """Vector store to retrieve documents for summarization"""
    
    def _run(self, instruction: str) -> str:
        """
        Generate summary based on instruction.
        
        Two modes:
        1. "all documents" → Summarize entire document collection
        2. Specific query → Summarize content related to query
        
        Args:
            instruction: Either "all documents" or a specific topic to summarize
            
        Returns:
            Generated summary string
        """
        try:
            if "all documents" in instruction.lower():
                # Mode 1: Summarize all uploaded documents
                # Retrieve many chunks to get broad coverage
                all_docs = self.vectorstore.similarity_search("", k=20)
                
                if not all_docs:
                    return "No documents available to summarize."
                
                # Use LangChain's map-reduce summarization chain
                # map_reduce is ideal for multiple documents:
                # - MAP: Summarize each document individually
                # - REDUCE: Combine individual summaries into final summary
                summarize_chain = load_summarize_chain(
                    self.llm,
                    chain_type="map_reduce"  # Handles documents longer than context window
                )
                
                summary = summarize_chain.run(all_docs)
                return f"Summary of all documents:\n\n{summary}"
            
            else:
                # Mode 2: Summarize content related to specific query
                # Retrieve relevant chunks for the query
                docs = self.vectorstore.similarity_search(instruction, k=5)
                
                if not docs:
                    return "No content found to summarize."
                
                # Combine retrieved content
                combined_content = "\n\n".join([d.page_content for d in docs])
                
                # Simple summarization using direct LLM call
                # For targeted summaries, we can use a simpler approach
                summary_prompt = f"Summarize the following content concisely:\n\n{combined_content[:3000]}"
                summary = self.llm.predict(summary_prompt)
                
                return summary
                
        except Exception as e:
            return f"Error creating summary: {str(e)}"
    
    async def _arun(self, instruction: str) -> str:
        """
        Async version for concurrent execution.
        
        Currently calls sync version.
        Future: Could implement async summarization for better performance.
        """
        return self._run(instruction)