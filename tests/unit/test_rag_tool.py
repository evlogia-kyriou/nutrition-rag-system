"""
Unit tests for RAG tool
"""

from src.agent.tools import NutritionRAGTool, MacroCalculatorTool

def test_rag_tool_initialization():
    """Test RAG tool can be initialized"""
    print("\n" + "="*60)
    print("TEST: RAG Tool Initializtion")
    print("="*60)

    try:
        tool = NutritionRAGTool()
        print(f" Tool initalized")
        print(f" Name: {tool.name}")
        print(f" Description : {tool.description[:80]}...")

        # Get tool info
        info = tool.get_tool_info()
        print(f"\n Tool Info:")
        for key, value in info.items():
            print(f" {key}: {value}")

        return True
    
    except Exception as e:
        print(f" Initialization faile: {e}")
        return False

def test_rag_tool_query():
    """Test RAG tool can answer queries"""
    print("\n" + "="*60)
    print("TEST : RAG Tool Query")
    print("="*60)

    try: 
        tool = NutritionRAGTool(similarity_top_k=3)

        # Test query
        query = "What is protein?"
        print(f"\n Query: {query}")

        result = tool._run(query)

        print("="*60)
        print("RESULT")
        print("="*60)
        print(result)
        print("="*60)

        if result and len(result) > 50:
            print(f" Query succesful")
            print(f" Response length: {len(result)} chars")
            return True
        else:
            print(f"\n Response too short or empty")
            return False
        
    except Exception as e:
        print(f" Query failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_calculator_tool():
    """Test calculator tool"""
    print("\n" + "="*60)
    print("TEST: Calculator Tool")
    print("="*60)

    try:
        tool = MacroCalculatorTool()

        # Test calculation
        query = "Calculate total calories from 50g protein, 200g carbs, 60g fat"
        print(f" Query: {query}")

        result = tool._run(query)

        print("\n" + "="*60)
        print("RESULT")
        print("="*60)
        print(result)
        print("="*60)

        if "Total Calories" in result:
            print(f"\n Calculation succesful")
            return True
        
        else :
            print(f" Calculation Failed")
            return False
    
    except Exception as e:
        print(f" Calculator test failed: {e}")
        return False
    
def main():
    """Run all tests"""
    print("="*60)
    print("RAG TOOL UNIT TESTS")
    print("="*60)

    results = [
        ("RAG Tool Initialization", test_rag_tool_initialization()),
        ("RAG Tool Query", test_rag_tool_query()),
        ("Calculator Tool", test_calculator_tool())
    ]

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = all(r for _, r in results)

    if all_passed:
        print("\n All test passed!")
    else :
        print("\n Some tests failed")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())



