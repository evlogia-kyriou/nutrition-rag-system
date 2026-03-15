"""
llama-cpp-python backend implementation.
Optimized for CPU inference with quantized models.
"""

import time
from typing import Dict, Any, Generator, Optional
from llama_cpp import Llama

from .base import BaseLLM, GenerationConfig, GenerationResult, BackendType

class LlamaCppLLM(BaseLLM):
    """
    llama-cpp-python backend for CPU-optimized inference
    Supports GGUF quantized models
    """

    def __init__(self, model_name: str, config: Dict[str, Any]):
        """Initialize llama-cpp backend
        
        Args:
            model_name: Patch to GGUF model file
            config: Configuration with keys:
                - model_path: Path to GGUF file
                - context_window: Context size (default: 8192)
                - n_threads: CPU threads (default: 8)
                - n_gpu_layers: GPU layers (default: 0 for CPU)
                """
        super().__init__(model_name, config)
        self.llm: Optional[Llama] = None

    def load_model(self) -> None:
        """Load GGUF model with llama-cpp-python"""
        if self._model_loaded:
            print(f"⚠️ Model already loaded")
            return
        
        model_path = self.config.get("model_path")
        if not model_path:
            raise ValueError("model_path required in config")
        
        print(f"📂 Loading model from: {model_path}")

        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=self.config.get("context_window", 8192),
                n_threads=self.config.get("n_threads", 8),
                n_gpu_layers=self.config.get("n_gpu_layers", 0),
                verbose=self.config.get("verbose", False),
                use_mlock=self.config.get("use_mlock", True),
                use_mmap=self.config.get("use_mmap", True),
            )

            self._model_loaded = True
            print(f"✅ llama-cpp-python model loaded successfully")

        except Exception as e:
            self._model_loaded = False
            raise RuntimeError(f"Failed to load model: {e}")
    
    def generate(
        self,
        prompt: str,
        generation_config: GenerationConfig
    ) -> GenerationResult:
        """
        Generate text synchronously with llama-cpp.
        
        Args:
            prompt: Input text
            generation_config: Generation parameters
            
        Returns:
            GenerationResult with response and metrics
        """
        if not self._model_loaded or self.llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        try:
            output = self.llm(
                prompt,
                max_tokens=generation_config.max_tokens,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                top_k=generation_config.top_k,
                stop=generation_config.stop_sequences or [],
                echo=False,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract response
            response_text= output["choices"][0]["text"]
            completion_tokens = output["usage"]["completion_tokens"]
            prompt_tokens = output["usage"]["prompt_tokens"]

            return GenerationResult(
                text=response_text,
                tokens_used=completion_tokens,
                latency_ms=latency_ms,
                model_name=self.model_name,
                backend="llama-cpp-python",
                metadata={
                    "prompt_tokens" : prompt_tokens,
                    "total_tokens": output["usage"]["total_tokens"],
                    "finish_reason": output["choices"][0].get("finish_reason", "unknown"),
                    "tokens_per_second": completion_tokens / (latency_ms/1000) if latency_ms > 0 else 0
                }
                
            )
        
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")
    
    def generate_stream(
            self,
            prompt: str,
            generation_config: GenerationConfig,
    ) -> Generator[str, None, None]:
        """
        Stream generation token by token
        
        Arg:
            prompt: Input text
            generation_config: Generation parameters
            
        Yields:
            Generated tokens
        """
        if not self._model_loaded or self.llm is None:
            raise RuntimeError ("Model not loaded. Call load_model() first.")
        
        try:
            stream = self.llm(
                prompt,
                max_tokens=generation_config.max_tokens,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                top_k=generation_config.top_k,
                stop=generation_config.stop_sequences or [],
                stream=True,
            )

            for output in stream:
                token = output["choices"][0]["text"]
                yield token

        except Exception as e:
            raise RuntimeError(f"Streaming failed: {e}")
        
    def unload_model(self) -> None:
        """Free model from memory"""
        if self.llm is not None:
            del self.llm
            self.llm = None
            self._model_loaded = False
            print(f"✅ Model unloaded from memory")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata"""
        return {
            "backend": "llama-cpp-python",
            "model_name": self.model_name,
            "model_path": self.config.get("model_path"),
            "context_window":self.config.get("context_window", 8192),
            "n_threads":self.config.get("n_threads",8),
            "n_gpu_layers": self.config.get("n_gpu_layers", 0),
            "device": "GPU" if self.config.get("n_gpu_layers", 0) > 0 else "CPU",
            "loaded": self._model_loaded 
        }

        
