import sys

def check_imports():
    """Verify all critical packages"""
    checks = {
        "Core Inference": [
            ("llama_cpp", "llama-cpp-python"),
            ("torch", "torch"),
        ],
        "LLM Frameworks": [
            ("llama_index.core", "llama-index-core"),
            ("llama_index.llms.llama_cpp", "llama-index-llms-llama-cpp"),
            ("llama_index.embeddings.huggingface", "llama-index-embeddings-huggingface"),
            ("langchain", "langchain"),
        ],
        "Vector Store": [
            ("chromadb", "chromadb"),
        ],
        "Phoenix":[
            ("phoenix", "arize-phoenix"),
            ("phoenix.otel", "arize-phoenix-otel"),
            ("openinference.instrumentation.langchain", "openinference-instrumentation-langchain"),
            ("openinference.instrumentation.llama_index", "openinference-instrumentation-llama-index"),
        ],
        "Document Processing":[
            ("fitz", "PyMuPDF"),
        ],
        "UI":[
            ("streamlit", "streamlit"),
            ("plotly", "plotly"),
        ],
        "Utilities":[
            ("yaml", "pyyaml"),
            ("dotenv", "python-dotenv"),
            ("pydantic", "pydantic"),
        ],

    }

    all_good = True

    for category, packages in checks.items():
        print(f"\n{'='*60}")
        print(f"📦{category}")
        print('='*60)

        for module, package in packages:
            try:
                imported = __import__(module)
                version = getattr(imported, "__version__","unknown")
                print(f"✅ {package:40} v{version}")
            except ImportError as e:
                print(f"❌ {package:40} MISSING")
                all_good = False
            except Exception as e:
                print(f"⚠️ {package:40} Error: {e}")
    return all_good

def check_bonus_packages():
    """Check bonus packages"""
    print(f"\n{'='}*60")
    print("🎁 BONUS PACKAGES")

    bonus = [
        ("langgraph", "LangGraph (advanced agents)"),
        ("langsmith", "LangSmith (debugging)"),
        ("fastapi", "FastAPI (API framework)"),
        ("tiktoken", "TikToken (token counting)"),
    ]

    for module, description in bonus:
        try:
            __import__(module)
            print(f"✅ {description}")
        except ImportError:
            print(f"⚠️ {description} - Not installed (optional)")

def test_llama_cpp():
    """Quick llama-cpp test"""
    print(f"\n{'n'*60}")
    print("🧪 LLAMA-CPP FUNCTIONALITY TEST")
    print('='*60)

    try:
        from llama_cpp import Llama
        print(" llama-cpp-python can be imported")
        print("    (Skipping model load test -  will do in next phase)")
        return True
    except Exception as e:
        print(f" llama-cpp-python import failed: {e}")
        return False
    
def main():
    print("=" * 60)
    print("NUTRTION RAG SYSTEM - SETUP VERIFICATION")
    print("=" * 60)
    
    results = []
    # Core packages
    print("\n🔍 Checking core packages...")
    core_ok = check_imports()
    results.append(("Core Packages", core_ok))

    # llama-cpp test
    cpp_ok = test_llama_cpp()
    results.append(("llama-cpp Test", cpp_ok))

    # Summary
    print("\n" + "="* 60)
    print("📊 SUMMARY")
    print("=" * 60)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")

    if all(r for _, r in results):
        print("\n" + "="*60)
        print("🎉 ALL CHECKS PASSED!")
        print("="*60)
        return 0
    else:
        print("\n⚠️ Some checks failed. Review errors above.")
        return 1
if __name__ == "__main__":
    sys.exit(main())


