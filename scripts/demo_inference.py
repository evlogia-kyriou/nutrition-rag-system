"""
Demo script showing how to use the inference abstraction layer
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
from src.inference import create_LLM, GenerationConfig

def main():
    """Demonstrate inference layer usage"""

    print("="*60)
    print("INFERENCE LAYER DEMO")
    print("="*60)

    # Load config

    with open("config/llm_config.yaml","r") as f:
        config = yaml.safe_load(f)

    
    # Create LLM using factory
    print("\n1️⃣ Creating LLM backend...")
    llm = create_LLM(
        backend="llamacpp",
        model_name="Llama-3.2-1B",
        config=config["backends"]["llamacpp"],
        auto_load=True
    )

    # Show model info
    print("\n2️⃣ Model Information:")
    info = llm.get_model_info()
    for key, value in info.items():
        print(f"    {key}: {value}")

    
    # Create generation config
    print("\n3️⃣ Setting up generation config...")
    gen_config = GenerationConfig(
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
    )

    # Generate response

    prompt = "What are the main macornutrients in food?"
    print(f"\n4️⃣ Generating response...")
    print(f"    Prompt: '{prompt}'")

    result = llm.generate(prompt, gen_config)

    print("="*60)
    print("RESULT")
    print("="*60)
    print(f"\n{result.text.strip()}\n")
    print("="*60)
    print(f"Tokens: {result.tokens_used}")
    print(f"Time: {result.latency_ms/1000:.2f}s")
    print(f"Speed: {result.metadata['tokens_per_second']:.1f} tok/s")
    print("="*60)

    # Cleanup
    llm.unload_model()
    print("\n Demo complete!")


if __name__ == "__main__":
    main()