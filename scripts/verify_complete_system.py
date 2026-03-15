"""
Complete system verification script.
Tests all phases (1-5) integration.
"""

import os
import sys
import yaml
import time
from pathlib import Path


def print_section(title):
    """Print section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def test_phase_1_environment():
    """Test Phase 1: Environment Setup"""
    print_section("PHASE 1: ENVIRONMENT SETUP")
    
    checks = []
    
    # Check Python packages
    print("\n1️⃣ Checking Python packages...")
    required_packages = [
        "llama_cpp",
        "llama_index.core",
        "langchain",
        "chromadb",
        "phoenix",
        "sentence_transformers",
        "streamlit",
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
            checks.append(True)
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            checks.append(False)
    
    # Check directory structure
    print("\n2️⃣ Checking directory structure...")
    required_dirs = [
        "src/inference",
        "src/data",
        "src/agent/tools",
        "data/raw",
        "data/vector_db",
        "models",
        "config",
    ]
    
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"   ✅ {dir_path}/")
            checks.append(True)
        else:
            print(f"   ❌ {dir_path}/ - MISSING")
            checks.append(False)
    
    # Check config files
    print("\n3️⃣ Checking configuration files...")
    config_files = [
        "config/llm_config.yaml",
        "config/rag_config.yaml",
        "config/phoenix_config.yaml",
    ]
    
    for file_path in config_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
            checks.append(True)
        else:
            print(f"   ❌ {file_path} - MISSING")
            checks.append(False)
    
    # Check model file
    print("\n4️⃣ Checking model file...")
    model_path = "./models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024**2)
        print(f"   ✅ Model found ({size_mb:.1f} MB)")
        checks.append(True)
    else:
        print(f"   ❌ Model not found at: {model_path}")
        checks.append(False)
    
    passed = all(checks)
    print(f"\n{'✅ PHASE 1 PASSED' if passed else '❌ PHASE 1 FAILED'}")
    return passed


def test_phase_2_inference():
    """Test Phase 2: Inference Layer"""
    print_section("PHASE 2: INFERENCE ABSTRACTION LAYER")
    
    try:
        print("\n1️⃣ Testing imports...")
        from src.inference import (
            create_llm,
            GenerationConfig,
            LLMFactory
        )
        print("   ✅ Imports successful")
        
        # Check backends registered
        print("\n2️⃣ Checking registered backends...")
        backends = LLMFactory.get_available_backends()
        print(f"   Available backends: {backends}")
        
        if "llamacpp" in backends:
            print("   ✅ llama-cpp backend registered")
        else:
            print("   ❌ llama-cpp backend NOT registered")
            return False
        
        # Test model loading
        print("\n3️⃣ Testing model loading...")
        with open("config/llm_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        llm = create_llm(
            backend="llamacpp",
            model_name="Llama-3.2-1B",
            config=config['backends']['llamacpp'],
            auto_load=True
        )
        print("   ✅ Model loaded successfully")
        
        # Test generation
        print("\n4️⃣ Testing text generation...")
        print("   ⏳ Generating (15-30 seconds)...")
        
        gen_config = GenerationConfig(
            max_tokens=30,
            temperature=0.7
        )
        
        result = llm.generate("Hello", gen_config)
        print(f"   ✅ Generation successful ({result.tokens_used} tokens)")
        print(f"   Response: {result.text[:50]}...")
        
        # Cleanup
        llm.unload_model()
        
        print("\n✅ PHASE 2 PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ PHASE 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase_3_data():
    """Test Phase 3: Data Processing"""
    print_section("PHASE 3: DATA PROCESSING LAYER")
    
    try:
        print("\n1️⃣ Testing imports...")
        from src.data import (
            NutritionIndexer,
            PDFDocumentLoader,
            VectorStoreManager
        )
        print("   ✅ Imports successful")
        
        # Check PDFs
        print("\n2️⃣ Checking PDF files...")
        pdf_dir = Path("./data/raw")
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        if pdf_files:
            print(f"   ✅ Found {len(pdf_files)} PDF files")
            for pdf in pdf_files:
                print(f"      • {pdf.name}")
        else:
            print("   ❌ No PDF files found in data/raw/")
            return False
        
        # Check vector database
        print("\n3️⃣ Checking vector database...")
        vector_db_path = Path("./data/vector_db")
        
        if vector_db_path.exists() and any(vector_db_path.iterdir()):
            print("   ✅ Vector database exists")
            
            # Check collection
            vsm = VectorStoreManager()
            stats = vsm.get_stats()
            print(f"   Documents indexed: {stats['document_count']}")
            
            if stats['document_count'] > 0:
                print("   ✅ Database has documents")
            else:
                print("   ⚠️  Database is empty - need to run indexing")
                return False
        else:
            print("   ❌ Vector database not found")
            print("   Run: python scripts/index_documents.py")
            return False
        
        # Test retrieval
        print("\n4️⃣ Testing retrieval...")
        indexer = NutritionIndexer()
        indexer.load_existing_index()
        
        query_engine = indexer.get_query_engine(similarity_top_k=3)
        response = query_engine.query("What is protein?")
        
        response_text = str(response)
        if response_text and len(response_text) > 50:
            print(f"   ✅ Retrieval successful ({len(response_text)} chars)")
            print(f"   Response preview: {response_text[:80]}...")
        else:
            print("   ⚠️  Retrieval response seems short")
        
        print("\n✅ PHASE 3 PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ PHASE 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase_4_tools():
    """Test Phase 4: RAG Tools"""
    print_section("PHASE 4: RAG TOOLS")
    
    try:
        print("\n1️⃣ Testing imports...")
        from src.agent.tools import (
            NutritionRAGTool,
            MacroCalculatorTool
        )
        print("   ✅ Imports successful")
        
        # Test RAG tool
        print("\n2️⃣ Testing RAG tool...")
        rag_tool = NutritionRAGTool(
            similarity_top_k=3,
            include_sources=True
        )
        print("   ✅ RAG tool initialized")
        
        # Test query
        print("\n3️⃣ Testing RAG query...")
        query = "What is protein?"
        result = rag_tool._run(query)
        
        if result and len(result) > 50:
            print(f"   ✅ Query successful ({len(result)} chars)")
            print(f"   Response preview: {result[:80]}...")
            
            if "Sources:" in result or "source" in result.lower():
                print("   ✅ Sources included")
            else:
                print("   ⚠️  Sources might be missing")
        else:
            print("   ❌ Query response too short")
            return False
        
        # Test calculator
        print("\n4️⃣ Testing calculator tool...")
        calc_tool = MacroCalculatorTool()
        calc_result = calc_tool._run(
            "Calculate total calories from 50g protein, 100g carbs, 30g fat"
        )
        
        if "Total Calories" in calc_result:
            print("   ✅ Calculator working")
            print(f"   Result: {calc_result[:80]}...")
        else:
            print("   ❌ Calculator not working correctly")
            return False
        
        print("\n✅ PHASE 4 PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ PHASE 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase_5_agent():
    """Test Phase 5: Agent"""
    print_section("PHASE 5: AGENT INTEGRATION")
    
    try:
        print("\n1️⃣ Testing imports...")
        from src.agent import NutritionAgent
        print("   ✅ Imports successful")
        
        # Initialize agent
        print("\n2️⃣ Initializing agent...")
        print("   ⏳ This will take a moment...")
        
        agent = NutritionAgent(
            backend="llamacpp",
            verbose=False,
            memory_window=5
        )
        print("   ✅ Agent initialized")
        
        # Get agent info
        info = agent.get_agent_info()
        print(f"\n   Agent configuration:")
        print(f"      Backend: {info['backend']}")
        print(f"      Tools: {', '.join(info['tools'])}")
        print(f"      Memory: {info['memory_window']} exchanges")
        
        # Test simple query
        print("\n3️⃣ Testing agent query...")
        print("   ⏳ Processing (30-60 seconds)...")
        
        query = "What is protein?"
        response = agent.query(query)
        
        if response and len(response) > 50:
            print(f"   ✅ Query successful ({len(response)} chars)")
            print(f"\n   Response preview:")
            print(f"   {response[:200]}...")
        else:
            print("   ❌ Query response too short or failed")
            return False
        
        # Test memory
        print("\n4️⃣ Testing conversation memory...")
        stats = agent.get_memory_stats()
        
        if stats['total_messages'] >= 2:  # At least question + answer
            print(f"   ✅ Memory tracking ({stats['total_messages']} messages)")
        else:
            print("   ⚠️  Memory might not be working")
        
        print("\n✅ PHASE 5 PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ PHASE 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run complete system verification"""
    
    print("="*60)
    print("  COMPLETE SYSTEM VERIFICATION")
    print("  Testing Phases 1-5")
    print("="*60)
    
    print("\n⚠️  This will take several minutes due to model loading")
    print("   and CPU inference time.")
    print("\nPress Ctrl+C to cancel, or Enter to continue...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        return
    
    # Track results
    results = []
    
    # Phase 1
    results.append(("Phase 1: Environment", test_phase_1_environment()))
    
    # Phase 2
    if results[0][1]:  # Only if Phase 1 passed
        results.append(("Phase 2: Inference", test_phase_2_inference()))
    else:
        print("\n⚠️  Skipping Phase 2 (Phase 1 failed)")
        results.append(("Phase 2: Inference", False))
    
    # Phase 3
    if results[0][1] and results[1][1]:  # Only if 1 & 2 passed
        results.append(("Phase 3: Data Processing", test_phase_3_data()))
    else:
        print("\n⚠️  Skipping Phase 3 (previous phases failed)")
        results.append(("Phase 3: Data Processing", False))
    
    # Phase 4
    if all(r[1] for r in results):  # Only if all previous passed
        results.append(("Phase 4: RAG Tools", test_phase_4_tools()))
    else:
        print("\n⚠️  Skipping Phase 4 (previous phases failed)")
        results.append(("Phase 4: RAG Tools", False))
    
    # Phase 5
    if all(r[1] for r in results):  # Only if all previous passed
        results.append(("Phase 5: Agent", test_phase_5_agent()))
    else:
        print("\n⚠️  Skipping Phase 5 (previous phases failed)")
        results.append(("Phase 5: Agent", False))
    
    # Final Summary
    print("\n\n" + "="*60)
    print("  VERIFICATION SUMMARY")
    print("="*60)
    
    for phase, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {phase}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL PHASES VERIFIED!")
        print("="*60)
        print("\n✅ Your system is fully functional!")
        print("✅ All integrations working correctly")
        print("✅ Ready to proceed to Phase 6")
        print("\n📝 Next steps:")
        print("   Tell me: 'Phase 5 complete, ready for Phase 6'")
        return 0
    else:
        print("⚠️  SOME PHASES FAILED")
        print("="*60)
        print("\n📝 Action items:")
        
        failed_phases = [name for name, passed in results if not passed]
        for phase in failed_phases:
            print(f"\n❌ {phase} needs attention:")
            
            if "Phase 1" in phase:
                print("   • Check package installations")
                print("   • Verify model downloaded")
                print("   • Check directory structure")
            
            elif "Phase 2" in phase:
                print("   • Review src/inference/ implementation")
                print("   • Test model loading manually")
                print("   • Check llama-cpp-python installation")
            
            elif "Phase 3" in phase:
                print("   • Add PDF files to data/raw/")
                print("   • Run: python scripts/index_documents.py")
                print("   • Check vector database creation")
            
            elif "Phase 4" in phase:
                print("   • Review src/agent/tools/ implementation")
                print("   • Ensure Phase 3 completed successfully")
                print("   • Test tools independently")
            
            elif "Phase 5" in phase:
                print("   • Review src/agent/langchain_agent.py")
                print("   • Ensure all previous phases passed")
                print("   • Check agent initialization")
        
        print("\n💡 Fix the failed phases, then run this script again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())