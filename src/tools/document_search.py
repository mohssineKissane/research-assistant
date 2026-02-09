"""
Document Search Tool - Semantic Search in Uploaded PDFs
========================================================

This tool enables the agent to search through uploaded PDF documents
using semantic similarity search (vector embeddings).

How the Agent Uses This Tool:
    The agent reads the tool's description and decides to use this tool when:
    - User asks about content in uploaded documents
    - Need specific quotes or data from PDFs
    - Looking for information that should be in the research papers

Tool Selection Example:
    User: "What does the paper say about transformers?"
    Agent Thought: This is asking about document content
    Agent Action: search_documents("transformers")
    
Why Tool Descriptions Matter:
    The description is the ONLY way the agent knows when to use this tool.
    A good description clearly states:
    - WHEN to use the tool (uploaded documents, specific info)
    - WHAT input it expects (search query)
    - WHAT output it provides (excerpts with citations)
"""
from langchain.tools import BaseTool
from pydantic import Field
from typing import Any

class DocumentSearchTool(BaseTool):
    """
    Tool for searching uploaded research documents using semantic similarity.
    
    This tool wraps the ChromaDB vector store to enable:
    - Semantic search (finds meaning, not just keywords)
    - Source attribution (returns page numbers and filenames)
    - Context retrieval (gets relevant excerpts for the agent)
    """
    
    name: str = "search_documents"
    """Tool identifier - the agent uses this name when calling the tool"""
    
    description: str = """
    Use this tool to search through the uploaded PDF documents.
    
    When to use:
    - User asks about content in uploaded documents
    - Need to find specific information from PDFs
    - Looking for quotes, data, or facts from research papers
    
    Input: A search query or question
    Output: Relevant excerpts from documents with source citations
    """
    """
    Critical: This description guides the agent's decision-making.
    The agent reads this to determine if this tool is appropriate for the query.
    """
    
    # Tool dependencies (injected at initialization)
    vectorstore: Any = Field(exclude=True)
    """ChromaDB instance for semantic search"""
    
    llm: Any = Field(exclude=True)
    """LLM instance (for potential future enhancements like re-ranking)"""
    
    def _run(self, query: str) -> str:
        """
        Execute document search and return formatted results.
        
        This is what gets called when the agent decides to use this tool.
        
        Process:
        1. Convert query to vector embedding
        2. Find k most similar document chunks in ChromaDB
        3. Format results with source citations
        4. Return formatted string to the agent
        
        Args:
            query: Search query from the agent
            
        Returns:
            Formatted string with search results and citations
        """
        try:
            # Retrieve top 4 most similar document chunks
            # ChromaDB automatically handles: query embedding → similarity search
            docs = self.vectorstore.similarity_search(query, k=4)
            
            if not docs:
                return "No relevant information found in uploaded documents."
            
            # Format results for the agent to read
            result = "Found the following information in documents:\n\n"
            
            for i, doc in enumerate(docs, 1):
                # Extract metadata for source attribution
                source = doc.metadata.get('filename', 'Unknown')
                page = doc.metadata.get('page', '?')
                content = doc.page_content[:300]  # First 300 chars to keep response concise
                
                result += f"[{i}] Source: {source}, Page: {page}\n"
                result += f"Content: {content}...\n\n"
            
            return result
            
        except Exception as e:
            # Return error message to agent (agent can decide to try another tool)
            return f"Error searching documents: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """
        Async version of _run for concurrent execution.
        
        Currently just calls the sync version.
        Future: Could implement true async for better performance.
        """
        return self._run(query)