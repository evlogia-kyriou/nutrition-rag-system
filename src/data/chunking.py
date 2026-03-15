"""
Test chunking strategy for document processing.
Implements sentence-window chunking for better context preservation
"""

from typing import List
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import Document
from llama_index.core.schema import TextNode

class ChunkingStrategy:
    """
    Manages document chunking strategies.
    Uses sentence-window approach for semantic boundaries
    """

    def __init__(
        self,
        window_size: int = 3,
        window_metadata_key: str = "window",
        original_text_metadata_key: str = "original_text"
    ):
        """
        Initialize chunking strategy.
        
        Args:
            window_size: Number of sentences before/after to include as context
            window_metadata_key: Metadata key for window text
            original_text_metadata_key: Metadata key for original sentence
            """
        self.window_size = window_size
        self.metada_key = window_metadata_key
        self.original_text_metadata_key = original_text_metadata_key

        print(f" Initializing sentence window chunker")
        print(f" Window size: {window_size} sentence")

        self.parser = SentenceWindowNodeParser.from_defaults(
            window_size = window_size,
            window_metadata_key = window_metadata_key,
            original_text_metadata_key=original_text_metadata_key,
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[TextNode] :
        """
        Chunk documents into nodes with sentence windows.
        Args: 
            documents: List of LlamaIndex documents
        
        Returns:
            List of text nodes with windowed context
        """
        print(f"\n Chungkin {len(documents)} documents...")

        nodes = self.parser.get_nodes_from_documents(
            documents,
            show_progress= True
        )

        print(f" Created {len(nodes)} chunks")

        # Statistics
        avg_chunk_size = sum(len(node.get_content()) for node in nodes) / len(nodes)
        print(f"    Avarage chunk size: {avg_chunk_size:.0f} characters")

        return nodes
    
    def get_chunk_stats(self, nodes: List[TextNode]) -> dict:
        """Get statistics about chunks"""

        chunk_sizes = [len(node.get_content()) for node in nodes]

        return {
            "total_chunks": len(nodes),
            "avg_size": sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
            "min_size": min(chunk_sizes) if chunk_sizes else 0,
            "max_size": max(chunk_sizes) if chunk_sizes else 0,
        }
