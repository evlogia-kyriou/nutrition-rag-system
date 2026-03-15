"""
Compare different retrieval configuration
Helps tune RAG parameters
"""
from src.agent.tools import NutritionRAGTool

def compare_top_k_values():
    """Compare different top_k values"""

    print("="*60)
    print("COMPARING TOP_K VALUES")
    print("="*60)

    query = "What is protein?"
    top_k_values = [1,3,5,10]
    for top_k in top_k_values:
        print(f"\n{'='*60}")
        print(f"Testing with top_k = {top_k}")
        print('='*60)

        tool = NutritionRAGTool(
            similarity_top_k = top_k,
            include_sources=True
        )

        result = tool._run(query)
        stats = tool.get_retrieval_stats(query)

        print(f"\nResponse length: {stats['response_length']} chars")
        print(f"Number of soures: {stats['num_sources']}")

        print(f"\nReponse preview:")
        print("-"*60)
        print(result[:300] + "...")
        print("="*60)

    
def compare_with_without_sources():
    """Compare response with and wihtout source citation"""

    print("\n" + "="*60)
    print("COMPARING WIHT/WITHOUT SOURCES")
    print("="*60)

    query = "What are vitamins?"

    # With sources
    print("\n 1. WITH SOURCES:")
    print("-"*60)
    tool_with = NutritionRAGTool(include_sources=True)
    result_with = tool_with._run(query)
    print(result_with)

    # Without source
    print("\n 2. WITHOUT SOURCES:")
    print("-"*60)
    tool_without = NutritionRAGTool(include_sources=False)
    result_without = tool_without._run(query)
    print(result_without)

def main():
    """Run comparisons"""
    compare_top_k_values()
    compare_with_without_sources()

    print("\n" + "="*60)
    print("Comparison Complete")
    print("\n"*60)

if __name__ == "__main__":
    main()
