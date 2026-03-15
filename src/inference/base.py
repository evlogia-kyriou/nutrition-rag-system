"""
Base classes and beta data structures for LLM inference abstraction.
Provides common interface for multiple inference backends.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Generator, List
from dataclasses import dataclass, field
from enum import Enum

class BackendType(Enum):
    """Available inference backends"""
    LLAMA_CPP = "llamacpp"
    VLLM = "vllm"

@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p : float = 0.9
    top_k : int = 40
    stop_sequences: Optional[List[str]] = None
    stream: bool = False

    def __post_init__(self):
        """Validate parameters"""
        if self.max_tokens <=0:
            raise ValueError("max_tokens must be positive")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
        if self.stop_sequences is None:
            self.stop_sequences = []

@dataclass
class GenerationResult:
    """Standardized result from any backend"""
    text: str
    tokens_used: int
    latency_ms: float
    model_name: str
    backend: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return (
            f"GenerationResult(\n"
            f"  text_preview='{self.text[:100]}...\n"
            f"  tokens={self.token_used}\n"
            f"  latency={self.latency_ms:.2f}ms\n"
            f"  backend={self.backend}\n"
            f")"
        )
    
class BaseLLM(ABC):
    """
    Abstract base class for LLM inference backends.
    All backends must implement these methods.
    """

    def __init__(self, model_name: str, config: Dict[str, any]):
        """
        Initialize backend with model and configuration
        
        Args:
            model_name= Model identifier (path or HF repo)
            config: Backend-specific configuraion dictionary
        """
        self.model_name = model_name
        self.config = config
        self.backend_name = self.__class__.__name__
        self._model_loaded = False

    @abstractmethod
    def load_model(self) -> None:
        """
        Load model into memory
        Should set self._model_loaded = True on success
        """
    
    @abstractmethod
    def generate(
        self,prompt: str, 
        generation_config: GenerationConfig
    ) -> GenerationResult:
        """
        Generate text synchrounously.

        Args:
            prompt: Input prompt text
            generation_config: Generation parameters
        
        Returns:
            GenerationResult with text and metadata

        Raises:
            RuntimeError: If model not loaded
        """
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        generation_config: GenerationConfig
    ) -> Generator[str, None, None]:
        """
        Stream generation token by token.
        Args:
            prompt: Input prompt text
            generation_config: Generation parameters

        Yields:
            Generated text tokens

        Raises:
            RuntimeError:If model not load
        """
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """
        Unload model from memory.
        Should set self._model_loaded =False
        """
        pass
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata and configuraion.

        Returns:
            Dictionary with model information
        """
        pass

    def is_loaded(self) -> bool:
        """
        Check if model is loaded
        """
        return self._model_loaded
    
    def __repr__(self):
        return f"{self.backend_name}(model={self.model_name}, loaded={self._model_loaded})"