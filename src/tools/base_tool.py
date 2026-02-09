from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field

class BaseResearchTool(BaseTool):
    """Base class for research tools"""
    
    # Tool metadata (used by agent)
    name: str = "base_tool"
    description: str = "Base tool description"
    
    def _run(self, query: str) -> str:
        """
        Synchronous implementation
        This is what gets called when agent uses the tool
        """
        raise NotImplementedError("Subclass must implement _run")
    
    async def _arun(self, query: str) -> str:
        """Async implementation (optional)"""
        raise NotImplementedError("Async not implemented")