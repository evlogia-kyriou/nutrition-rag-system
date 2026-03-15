"""
Inference abstraction layer.
Provides unified interface for multiple LLM backends.
"""
from .base import BaseLLM, GenerationConfig, GenerationResult, BackendType
from .factory import LLMFactory, create_LLM
from .llama_cpp_backend import LlamaCppLLM

# Register llama-cpp backend
LLMFactory.register_backend(BackendType.LLAMA_CPP, LlamaCppLLM)

__all__=[
    "BaseLLM",
    "GenerationConfig",
    "GenerationResult",
    "BackendType",
    "LLMFactory",
    "create_LLM",
    "LlamaCppLLM",
]