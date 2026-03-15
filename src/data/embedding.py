"""
Embedding model manager.
Handles text embedding generation for vector storage.
"""

from typing import List
import torch
from sentence_transformers import SentenceTransformer

class EmbeddingManager:
    """,

    Manages embedding model for text vectorization.
    Uses sentence-transformers for local embedding generation
    """
    def __init__(
        self,
        model_name: str = "sentence-transformer/all-MiniLM-L6-v2",
        device: str= "cpu",
        batch_size: int = 32

    ):
        """
        Initialize embedding model.
        
        Args:
            model_name: HuggingFace model identifier
            device: "cpu" or "cuda"
            batch_size: Batch size for embedding generation
        """
        self.model_name= model_name
        self.device = device
        self.batch_size = batch_size

        print(f" Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        print(f"✅ Embedding model loaded")
        print(f" Dimensions: {self.model.get_sentence_embedding_dimension()}")
        print(f" Device: {device}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for single text.
        Arg:
            test: Input text
        
        Returns:
            Embedding vector as list of floats
        
        """
        embedding = self.model.encode(text, conver_to_tensor=False)
        return embedding.tolist()
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()
    
    def get_model_info(self) -> dict:
        """Get model metadata"""
        return{
            "model_name": self.model_name,
            "dimensions": self.get_dimension(),
            "device":self.device,
            "batch_size": self.batch_size,
            "max_seq_length": self.model.max_seq_length
        }