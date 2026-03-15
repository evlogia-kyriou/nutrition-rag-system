"""
Factory pattern for creating LLM backend instances.
Handles backend selection and instantiation.
"""

from typing import Dict, Any, Optional
from .base import BaseLLM, BackendType

class LLMFactory:
    """
    Factory for creating LLM backend instances.
    Supports multiple backends through unified interface.
    """

    # Registry of available backends
    _backends : Dict[str, type] = {}

    @classmethod
    def register_backend(cls, backend_type: BackendType, backend_class: type):
        """
        Register a new backend implementation
        
        Arg:
            backend_type: Backend identifier
            backend_class: Class implementing BaseLLM
        """
        if not issubclass(backend_class, BaseLLM):
            raise TypeError(f"{backend_class} must inherit from BaseLLM")
        
        cls._backends[backend_type.value] = backend_class
        print(f" Registered backend: {backend_type} -> {backend_class.__name__}")

    @classmethod
    def create(
        cls,
        backend: str,
        model_name: str,
        config: Dict[str, Any],
        auto_load: bool = True
    ) -> BaseLLM:
        """
        Create and optionally load an LLM backend instance.

        Args:
            backend: Backend type ("llamacpp" or "vllm")
            model_name: Model identifier
            config: Backend-specific configuration
            auto_load: Whether to automatically load the model

        Returns:
            Initialized LLM backend instance

        Raises:
            ValueError: If backend not supported
        """
        if backend not in cls._backends:
            available = list(cls._backends.keys())
            raise ValueError(
                f"Unsupported backend: '{backend}'"
                f"Available backends: {available}"
            )
        
        # Gete backend class and instantiate
        backend_class = cls._backends[backend]
        instance = backend_class(model_name,config)

        # Auto-load model if requested
        if auto_load:
            print(f"🔄 Loading model with {backend}...")
            instance.load_model()
            print(f"✅ Model loaded successfully")
        
        return instance
    
    @classmethod
    def get_available_backends(cls) -> list:
        """Get list of registered backend names"""
        return list(cls._backends.keys())
    
    @classmethod
    def is_backend_available(cls, backend: str) -> bool:
        """Check if a backend is registered"""
        return backend in cls._backends

# Conveniene function
def create_LLM(
        backend:str,
        model_name: str,
        config: Dict[str, Any],
        auto_load: bool =True
) -> BaseLLM:
    """
    Convenience function to create LLM Backend

    Arg:
        backend: Backend type
        model_name: Model identifier
        config: Backend configuration
        auto_load: Whether to load model immediately

    Return:
        LLM backend instance
    """

    return LLMFactory.create(backend, model_name, config, auto_load)


