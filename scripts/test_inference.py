from llama_cpp import Llama
import time

def test_basic_inference():
    """Test basic model loading and generation"""
    model_path = "./models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

    print("Loading model...")
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=8,
        verbose=False
    ) 

    print("✅ Model Loaded")

    prompt = "What is protein and why is it important"

    print(f"\nPrompt: {prompt}")
    print("\nGenerating response...")

    start_time= time.time()

    output= llm(
        prompt, 
        max_tokens=100,
        temperature=0.7,
        stop=["User:", "\n\n"],
    )

    latency = time.time() - start_time

    response = output["choices"][0]["text"]
    tokens = output["usage"]["completion_tokens"]

    print(f"\nResponse: {response}")
    print(f"\n 📊Metrics:")
    print(f"    Latency: {latency:.2f} seconds")
    print(f"    Tokens: {tokens}")
    print(f"    Speed: {tokens/latency:.1f} tokens/sec")

    return True

if __name__ == "__main__":
    test_basic_inference()