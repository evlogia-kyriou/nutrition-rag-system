"""
Script to index nutrition PDF documents.
Run once to create the vector database.
"""

import argparse
from src.data.indexing import NutritionIndexer

def main():
    """Index Nutrition documents"""

    parser = argparse.ArgumentParser(
        description="Index nutrition PDF documents into vector database"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reindex even if index exists"
    )
    parser.add_argument(
        "--pdf-dir",
        default="./data/raw",
        help="Directory containg PDF files"
    )
    parser.add_argumetn(
        "--test-query",
        type=str,
        help="Test query after indexing"
    )

    args = parser.parse_args()

    indexer = NutritionIndexer(
        pdf_directory=args.pdf_dir,
        persist_directory="./data/vector_db",
        collection_name="nutrition_docs"
    )
        
    # Create or load index

    try:
        indexer.create_index(force_reindex=args.force)

        # Test query if provided
        if args.test_query:
            indexer.test_query(args.test_query)
        else:
            # Default test query
            print("\n" + "="*60)
            print("Running default test query...")
            print("="*60)
            indexer.test_query("Where is protein and why is it important?")

        print("\n" + "="*60)
        print("INDEXING COMPLETE!")
        print("="*60)
        print("\n Summary:")
        print(f" Vector DB: ./data/vector_db")
        print(f" Collection: nutrition_docs")
    
    except FileNotFoundError as e:
        print(f"\n Error: {e}")
        print("\n Make sure to:")
        print(" 1. Place PDF files in data/raw/")
        print(" 2. Check that PDF files exist")
        print(f"\n Looking in: {args.pdf_dir}")
    
    except Exception as e:
        print(f"\n Indexing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


