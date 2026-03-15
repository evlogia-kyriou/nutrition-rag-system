"""
ChromaDB vector store wrapper.
Manages persistent vector storage for document embeddings.
"""

import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import Optional

class VectorStoreManager:
    """
    Manages ChromaDB vector store for document embeddings
    Provides persistent storages with easy initialization.
    """

    def __init__(
            self,
            persist_directory: str = "./data/vector_db",
            collection_name: str = "nutrition_docs"
    ):
        """
        Initialize vector store.
        
        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the collection
        """

        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name

        # Create directory if needed
        self.persist_directory.mkdir(parents=True, exist_ok= True)

        print(f" Initializing ChromaDB")
        print(f" Localization: {self.persist_directory}")
        print(f" Collection: {collection_name}")

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hhnsw:space": "cosine"} # Use cosine similarity
        )

        print(f" ChromaDB initialized")
        print(f" Existing documents: {self.collection.count()}")

    def get_collection(self):
        """Get ChromaDB collection"""
        return self.collection
    
    def get_client(self):
        """Get ChromaDB client"""
        return self.client
    
    def collection_exists(self) -> bool:
        """Check if collection has documents"""
        return self.collection.count() > 0
    
    def get_stats(self) -> dict:
        """Get vector store statistics"""
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "persist_directory": self(self.persist_directory),
        }
    
    def clear_collection(self):
        """Delete all documents from collection"""
        print(f" Clearing collection: {self.collection_name}")
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f" Collection cleared")