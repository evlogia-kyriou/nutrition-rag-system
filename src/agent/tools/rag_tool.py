"""
RAG tool for retrieving nutrition information from vector store.
Queries indexed documents and returns relevant passages.
"""

from typing import Optional, List, Dict, Any
from pydantic import Field

from src.data.indexing import NutritionIndexer
from .base_tool import BaseAgentTool


class NutritionRAGTool(BaseAgentTool):
    """
    Tool for retrieving nutrition information from indexed documents.
    Uses Llamaindex query engine to find relevant passages
    """

    name: str = "nutrition_rag"
    description: str = (
        "Use this tool to search nutrition information from autoritative sourves."
        "Input should be a specific question about nutrition, food, vitamins,"
        "macronutrients, health, or diet. The tool returns relevant information "
        "from nutrition textbooks and research."
    )

    # Tool-specific fields
    indexer: Optional[NutritionIndexer] = Field(default=None, exclude=True)
    similarity_top_k: int = Field(default=5, description="Number of chunks to retrieve")
    include_sources: bool = Field(default=True, description="Include source citation")

    def __init__(
        self,
        similarity_top_k: int = 5,
        include_sources: bool = True,
        **kwargs

    ):
        """
        Initialize RAG tool.
        
        Args:
            similarity_top_k: Number of similar chunks to retrieve
            include_sources: Whether to include source citations
        """

        # Initialize parent
        super().__init__(
            similarity_top_k=similarity_top_k,
            include_sources=include_sources,
            **kwargs
        )
        
        # Load indexer
        print(f" Initializing RAG tool...")
        self.indexer = NutritionIndexer()
        self.indexer.load_existing_index()
        print(f" RAG tool ready")

    def _run(self, query: str) -> str:
        """
        Execute RAG retrieval.
        
        
        Args:
            query: Nutrition question
            
        Returns:
            Formatted response with retrieved information
        """
        if not query or not query.strip():
            return "Error: Query cannot be empty"
        
        try:
            # Query the index
            query_engine = self.indexer.get_query_engine(
                similarity_top_k=self.similarity_top_k
            )
            response = query_engine.query(query)

            # Format response
            result = self._format_response(response)
            return result
        
        except Exception as e:
            return f"Error retrieving information: {str(e)}"
        
    def _format_response(self, response) -> str:
        """
        Format query response for agent consumption
        
        Args:
            response: LlamaIndex query response
            
        Returns:
            Formatted string with anser and sources
        """
        # Get main response text
        answer = str(response).strip()

        # Add source information if requested
        if self.include_sources and hasattr(response, 'source_nodes'):
            sources = self._format_sources(response.source_nodes)
            if sources:
                answer += f"\n\n{sources}"
        
        return answer
    
    def _format_sources(self, source_nodes) -> str:
        """
        Format source citations.
        Args: 
            source_node: List of source nodes from response
        
        Returns:
            Formatted source string
        """
        if not source_nodes:
            return ""
        
        sources_text ="Sources:"
        for i, node in enumerate(source_nodes[:3], 1):
            book_name = node.metadata.get('book_name', 'Unknown')
            page_num = node.metadata.get('page_number', 'Unknown')
            score = getattr(node, 'score', 0.0)

            sources_text += (
                f"\n{i}. {book_name} (Page {page_num})"
                f"[Relevance: {score:.2f}]"
            )
        return sources_text
    
    def get_retrieval_stats(self, query: str) -> Dict[str, Any]:
        """
        Get detailed retrieval statistics for a query.
        Useful for debugging and monitoring.
        
        Args:
            query: Search query
        
        Returns:
            Dictionary with retrieval statistics
        """
        query_engine = self.indexer.get_query_engine(
            similarity_top_k=self.similarity_top_k
        )
        response = query_engine.query(query)

        stats = {
            "query" : query,
            "response_length" : len(str(response)),
            "num_sources" : len(response.source_nodes) if hasattr(response, 'source_nodes') else 0,
        }

        if hasattr(response, 'source_nodes'):
            stats["source_details"] = [
                {
                    "book" : node.metadata.get('book_name'),
                    "page" : node.metadata.get('page_number'),
                    "score" : getattr(node, 'score', '0.0'),
                    "text_length" : len(node.text)
                }
                for node in  response.source_nodes
            ]
        return stats

class NutritionRAGToolWithContext(NutritionRAGTool):
    """
    Enhanced RAG took that includes chunk cotnext in response.
    Provides more detailed information for compex queries.
    """

    name: str = "nutrition_rag_detailed"
    description: str = (
        "Use this tool for detailed nutrition information with full context."
        "Better for complex queries requiring comprehensive answers."
        "Input shoud be a specific question about nutrition."   
        )
    
    max_context_length: int = Field(default=1500, description="Max context characters")

    def _format_response(self, response) -> str:
        """
        Format response with additional context from chunks.
        
        Args:
            response: LlamaIndex query reponse
            
        Returns:
            Formatted string with answer, context, and sources
        """
        answer = str(response).strip
        
        # Add relevant chunks as context
        if hasattr(response, 'source_nodes') and response.source_nodes:
            context = self._format_context(response.source_nodes)
            if context:
                answer += f"\n\nRelevant Context:\n{context}"
            
        # Add sources
        if self.include_sources and hasattr(response, 'source_nodes'):
            sources = self._format_sources(response.source_nodes)
            if sources:
                answer += f"\n\n{sources}"
        
        return answer
    
    def _format_context(self, source_nodes) -> str :
        """
        Format context from source chunks.
        Args:
            source_nodes: List of source nodes
            
        Returns:
            Formatted context string
        """
        context_parts = []
        total_length = 0

        for i, node in enumerate(source_nodes[:3],1):
            chunk_text = node.text[:500].strip # First 500 chars

            if total_length + len(chunk_text) > self.max_context_length:
                break

            context_parts.append(f"[{i}] {chunk_text}...")
            total_length += len(chunk_text)

        return "\n\n.join(contex_parts)"
    




