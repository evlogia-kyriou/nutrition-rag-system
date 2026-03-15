"""
Unit tests for llama-cpp backend
Test basic functionality without heavy model laoding
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
import pytest
from src.inference import (
    LlamaCppLLM,
    GenerationConfig,
    BackendType
)

def test_backend_initialization():
    """Test backend can be initialized"""
    config = {
        "model_path" : "./models/test.gguf",
        "context_window" : 2048,
        "n_threads" : 4,
    }

    backend = LlamaCppLLM(
        model_name="test-model",
        config=config
    )

    assert backend.model_name == "test-model"
    assert backend.config["context_window"] == 2048
    assert not backend.is_loaded()
    print("✅ Backend initialization test passed")

def test_generation_config_validation():
    """Test generation config validates parameters"""

    # Valid config

    config = GenerationConfig(
        max_tokens=100,
        temperature=0.7,
        top_p=0.9
    )

    assert config.max_tokens == 100

    # Invalid temperature
    with pytest.raises(ValueError):
        GenerationConfig(temperature=3.0)

    # Invalid top_p

    with pytest.raises(ValueError):
        GenerationConfig(top_p=1.5)
    
    print("✅ Generation config validation test passed")

def test_model_info():
    """Test model info retrieval"""
    config={
        "model_path" : "./models/test.gguf",
        "context_window" : 4096,
        "n_threads" : 8,
        "n_gpu_layers" : 0,
    }

    backend = LlamaCppLLM("test-model", config)
    info = backend.get_model_info()

    assert info["backend"] == "llama-cpp-python"
    assert info["context_window"] == 4096
    assert info["device"] == "CPU"
    assert not info["loaded"]

    print("✅ Model info test passed")

if __name__ == "__main__":
    test_backend_initialization()
    test_generation_config_validation()
    test_model_info()
    print("\n🎉 All unit test passed")
