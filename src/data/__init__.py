"""
Data processing layer for document ingestion and indexing
"""

from .document_loader import PDFDocumentLoader
from .chunking import ChunkingStrategy
from .vector_store import VectorStoreManager
from .indexing import NutritionIndexer
from .embedding import EmbeddingManager

__all__ = [
    "PDFDocumentLoader",
    "ChunkingStrategy",
    "VectorStoreManager",
    "NutritionIndexer",
    "EmbeddingManager",
]