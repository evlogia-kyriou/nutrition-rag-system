"""
Main indexing pipeline.
Orchestrates document loading, chunking, embedding, and storage.
"""

from typing import Optional, List
from pathlib import Path
import yaml

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from .document_loader import PDFDocumentLoader
from .chunking import ChunkingStrategy
from .vector_store import VectorStoreManager

class NutritionIndexer:
    """
    Complete idexing pipeline for nutrition documents.
    Handles loading, chunking, embedding, and vector storage.
    """

    def __init__(
        self,
        pdf_directory: str = "./data.raw",
        persist_directory: str = "./data/vector_db",
        collection_name: str = "nutrition_docs",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        window_size: int = 3
    ):
        """
        Initialize indexing pipeline.
        args:
            pdf_directory: Directory containing PDF files
            persist_directory: Directory for vector storage
            collection_name: ChromaDB collection name
            embedding_model: HuggingFace embedding model
            window_size: Sentence window size for chungking
        """
        self.pdf_directory = pdf_directory
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        print("="*60)
        print("NUTRITION DOCUMENT INDEXER")
        print("="*60)

        #Initialize components
        print("\n Initializing components...")

        # Setup embedding model for LlamaIndex
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model,
            device="cpu"
        )
        print(f"    Embedding model: {embedding_model}")

        # Initialize vector store
        self.vector_store_manager = VectorStoreManager(
            persist_directory=persist_directory,
            collection_name=collection_name
        )

        # Initialize document loader
        self.doc_loader = PDFDocumentLoader(pdf_directory=pdf_directory)

        # Initialize chunking strategy
        self.chunking_strategy = ChunkingStrategy(window_size=window_size)

        self.index: Optional[VectorStoreIndex] = None
    
    def create_index(self, force_reindex: bool = False) -> VectorStoreIndex:
        """
        Create or load vector index.
        
        Args:
            force_reindex: If True, recreate index even if it exists
        Returns:
            VectorStoreIndex instance
        """
        # Check if index already exists
        if not force_reindex and self.vector_store_manager.collection_exists():
            print("\n Index already exists!")
            print(" Use force_reindex=True to recreate")
            return self.load_existing_index()
        
        if force_reindex and self.vector_store_manager.collection_exists():
            print("\n Forcing reindex - clearing existing data")
            self.vector_store_manager.clear_collection()
        
        # Load documents
        print("\n Loading PDF Documents...")
        documents = self.doc_loader.load_all_pdf()

        # Get document stats
        stats = self.doc_loader.get_document_stats(documents)
        print(f"\n Document Statistics:")
        for book_name, book_stats in stats["books"].items():
            print(f"    •{book_name}: {book_stats['pages']} pages,"
                  f"{book_stats['chars']:,} chars")
        
        # Chunk documents
        print("\n Chunking documents...")
        nodes = self.chunking_strategy.chunk_documents(documents)

        chunk_stats = self.chunking_strategy.get_chunk_stats(nodes)
        print(f"\n Chunk Statistics:")
        print(f"    • Total chunks: {chunk_stats['total_chunks']}")
        print(f"    • Avg size: {chunk_stats['avg_size']:.0f} chars")
        print(f"    • Min size: {chunk_stats['min_size']} chars")
        print(f"    • Max size: {chunk_stats['max_size']} chars")

        # Create vector store
        print("\n Creating vector index...")
        print(" Generating embeddings")

        # Setup ChromaDb vector store
        chroma_collection = self.vector_store_manager.get_collection()
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create index from nodes
        self.index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            show_progress=True
        )

        print(f"\n Index created successfully!")
        print(f" Documents: {len(documents)}")
        print(f" Chunks: {len(nodes)}")
        print(f" Stored in: {self.persist_directory}")

        return self.index
    
    def load_existing_index(self) -> VectorStoreIndex:
        """
        Load existing index from disk.
        Returns:
            VectorStoreIndex instance
        """
        print("\n Loading existing index...")

        # Setup ChromaDB vector store
        chroma_collection = self.vector_store_manager.get_collection()
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # Load index
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store
        )

        stats = self.vector_store_manager.get_stats()
        print(f" Index loaded")
        print(f" Documents: {stats['document_count']}")

        return self.index
    
    def get_query_engine(self, similarity_top_k: int = 5):
        """
        Get query engine for retrieval.
        
        Args:
            similarity_top_k: Number of similar chunks to retrieve
        
        Returns:
            Query engine
        """

        if self.index is None:
            raise RuntimeError("Index not created. Call create_index() first.")
        
        return self.index.as_query_engine(similarity_top_k=similarity_top_k)
    
    def test_query(self, query: str, similarity_top_k: int = 3):
        """
        Test query on index.
        
        Args:
            query: Query text
            similarity_top_k: Number of results to retrieve
        """
        print(f"\n Testing query: '{query}'")
        print(f" Retrieving top {similarity_top_k} results...\n")

        query_engine = self.get_query_engine(similarity_top_k=similarity_top_k)
        response = query_engine.query(query)

        print("="*60)
        print("RESPONSE")
        print(f"{response}")

        if hasattr(response, 'source_nodes'):
            print("="*60)
            print("SOURCES")
            print("="*60)
            for i, node in enumerate(response.source_nodes, 1):
                print(f"\n{i}. Score: {node.score:.4f}")
                print(f" Book: {node.metadata.get('book_name','Unknown')}")
                print(f" Page: {node.metadata.get('page_number', 'Unknown')}")
                print(f" Text preview: {node.text[:200]}...")






