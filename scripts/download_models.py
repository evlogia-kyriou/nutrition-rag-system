from huggingface_hub import hf_hub_download
import os

def download_llama_model():
    """Download Llama 3.2 1B GGUF model"""
    model_name = "bartowski/Llama-3.2-1B-Instruct-GGUF"
    filename = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"

    print(f"Downloading {filename}...")

    try:
        model_path = hf_hub_download(
            repo_id=model_name,
            filename=filename,
            local_dir="./models",
            local_dir_use_symlinks=False
        )

        print(f" Model download to: {model_path}")

        return model_path
    
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

if __name__ == "__main__":
    model_path = download_llama_model()
    
    if model_path and os.path.exists(model_path):
        print("\n✅ Setup Complete!")
        print(f"Model location: {model_path}")
    else:
        print("\n❌ Download incomplate")
