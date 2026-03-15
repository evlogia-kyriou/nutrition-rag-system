"""
Integration test for llama-cpp backend with actual model.
This will load real model and test generation
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import os
import yaml
from src.inference import create_LLM, GenerationConfig

def load_config():
    """Load configuration from YAML"""
    with open ("config/llm_config.yaml","r") as f :
        return yaml.safe_load(f)
    
def test_model_loading():
    """Test that model can be loaded"""
    print("\n" + "="*60)
    print("Test 1: Model Loading")
    print("="*60)

    config = load_config()
    model_path = config["backends"]["llamacpp"]["model_path"]

    # Check model file exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print(" Run: python scripts/download_models.py")
        return False
    
    print(f"✅Model file found: {model_path}")
    print(f"  Size: {os.path.getsize(model_path) / (1024**2):.1f} MB")

# Create backend
    try:
        llm = create_LLM(
            backend="llamacpp",
            model_name="Llama-3.2-1B",
            config=config["backends"]["llamacpp"],
            auto_load=True
        )

        print(f"✅ Model loaded successfully")
        print(f"    Backend: {llm.backend_name}")
        print(f"    Loaded: {llm.is_loaded()}")

        # Get model info
        info = llm.get_model_info()
        print(f"\n📊 Model Info:")
        for key, value in info.items():
            print(f"    {key}: {value}")
        
        llm.unload_model()
        return True
    
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_simple_generation():
    """Test basic text generation"""
    print("\n" + "=" * 60)
    print("Test 2: Simple Generation")
    print("=" * 60)

    config = load_config()

    # Create_backend
    llm = create_LLM(
        backend="llamacpp",
        model_name="Llama-3.2-1B",
        config=config["backends"]["llamacpp"],
        auto_load=True
    )

    # Create generation config
    gen_config = GenerationConfig(
        max_tokens=50,
        temperature=0.7,
        top_p=0.9,
    )

    print(f"    Loaded: {llm.is_loaded()}")

    # Test prompt
    prompt = "What is protein"
    print(f"\n Prompt: {prompt}")
    print(f" Generating...")

    try:
        result = llm.generate(prompt, gen_config)

        print(f"✅ Generation successful!")
        print(f"\n📝 Response:")
        print(f"    {result.text.strip()}")
        print(f"\n📊 Metrics:")
        print(f"    Tokens: {result.tokens_used}")
        print(f"    Latency: {result.latency_ms/1000:.2f}s")
        print(f"    Speed: {result.metadata['tokens_per_second']:.1f} tokens/sec")
        print(f"    Backend: {result.backend}")

        llm.unload_model()
        return True
    
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        llm.unload_model()
        return False

def test_streaming_generation():
    """Test streaming generation"""
    print("\n" + "=" * 60)
    print("Test 3: Streaming Generation")
    print("=" * 60)

    config = load_config()

    llm = create_LLM(
        backend="llamacpp",
        model_name="Llama-3.2-1B",
        config=config["backends"]["llamacpp"],
        auto_load=True
    )

    gen_config = GenerationConfig(
        max_tokens=30,
        temperature=0.7,
        stream=True,
    )

    prompt = "List 3 benefits of protein:"
    print(f"\n Prompt: {prompt}")
    print(f" Streaming...\n")
    print(f" Response: ", end="", flush= True)

    try:
        for token in llm.generate_stream(prompt, gen_config):
            print(token, end="", flush=True)
        
        print(f"\n\n Streaming successful!")

        llm.unload_model()
        return True
    
    except Exception as e:
        print(f"\n Streaming failed: {e}")
        llm.unload_model()
        return False
    
def main():
    """Run all integration tests"""
    print("="*60)
    print("LLAMA-CPP BACKEND INTEGRATION TESTS")
    print("="*60)

    results = []

    # Test 1 : Loading
    results.append(("Model Loading", test_model_loading()))

    # Test 2 : Generation
    if results[0][1]: # Only if loading work
        results.append(("Simple Generation", test_simple_generation()))
        results.append(("Streaming Generation", test_streaming_generation()))

    # Summary
    print("\n" + "="*60)
    print(" ALL TESTS PASSED!")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(r for _, r in results)
    if all_passed:
        print("\n llama-cpp backend is working correctly")
        print(" Model can generate text")
        print(" Streaming works")
        print("\n Ready to proceed to Phase3!")
        print(" Model can generate text")
        print(" Streaming works")
    
    else:
        print("\n Some tests failed. Review errors above")

    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
