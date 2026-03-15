"""
Test script for document retrieval
"""

from src.data.indexing import NutritionIndexer

def test_retrieval():
    """Test retrieval with various queries"""

    print("="*60)
    print("TESTING DOCUMENT RETRIEVAL")
    print("="*60)

    # Load existing index
    indexer = NutritionIndexer()
    indexer.load_existing_index()

    # Test queries
    test_queries = [
        "What is protein",
        "What are the benefits or vitamins?",
        "How much water should I drink daily",
        "What foods are high in fiber",
        "What are carbohydrates?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}/{len(test_queries)}")
        print('='*60)
        indexer, test_query(query, similarity_top_k=3)

        if i < len(test_queries):
            input("\nPress Enter for next query...")

    print("\n" + "="*60)
    print("All test queries complete!")
    print("="*60)

if __name__ == "__main__":
    test_retrieval()