"""
Base tool interface for agent tools.
Provides common structure for all tools
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from langchain.tools import BaseTool as LangChainBaseTool
from pydantic import Field

class BaseAgentTool(LangChainBaseTool, ABC):
    """
    Base class for all agent tools.
    Extends LangChain BaseTool with common functionality.
    """

    # Metadata that subclasses should override
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description for agent")

    def __init__(self, **kwargs):
        """Initialize tool with validation"""
        super().__init__(**kwargs)
        self._validate_tool()
    
    def _validate_tool(self):
        """Validate tool configuration"""
        if not self.name:
            raise ValueError("Tool must have a name")
        if not self.description:
            raise ValueError("Tool must have a description")
        
    @abstractmethod
    def _run(self, query:str) -> str:
        """
        Execute tool logic.
        Args:
            query: Input query string
        Returns:
            Tool result as string
        """
        pass

    async def _arun(self, query: str) -> str:
        """
        Async execution (optional).
        Default implementation calls sync version
        """
        return self._run(query)
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get tool metadata"""
        return{
            "name": self.name,
            "desciption": self.description,
            "type": self.__class__.__name__
        }