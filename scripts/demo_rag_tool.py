"""
Interactive demo of RAG tool.
Shows tool capabilities with various queries
"""

from src.agent.tools import NutritionRAGTool, MacroCalculatorTool

def demo_rag_tool():
    """Demonstrate RAG tool sampe queries"""

    print("="*60)
    print("RAG TOOL INTERACTIVE DEMO")
    print("="*60)

    # Iniatialize tool
    print("\n Initializing RAG tool...")
    rag_tool = NutritionRAGTool(
        similarity_top_k=3,
        include_sources=True
    )

    print("Ready!\n")

    # Sample queries

    queries = [
        "What is protein and why is it important?"
        "What are the main sources of vitamin C?"
        "How many calories are in carbohydrates?"
        "What is the recommended daily water intake?"
        "What foods are high in fiber?"
    ]

    for i, query in enumerate(queries, 1):
        print("="*60)
        print(f"Query  {i}/{len(queries)}")
        print("="*60)
        print(f"Question: {query}\n")

        result = rag_tool._run(query)

        print("Answer :")
        print("-"*60)
        print(result)

        if i < len(queries):
            input("\nPress Enter for next query...")
            print()

def demo_calculator_tool():
    """Demonstrate calculator tool"""

    print("\n" + "="*60)
    print("CALCULATOR TOOL DEMO")
    print("="*60)

    calc_tool = MacroCalculatorTool()

    calculations = [
        "Calculate total calories 50g protein, 200g carbs, 60g fat",
        "What's is the macro percentage of 150g protein in 2000 calories?",
        "Calculate: 100 * 4 + 50 * 9",
    ] 

    for i, calc in enumerate(calculations, 1):
        print(f"\n{i}, {calc}")
        result = calc_tool._run(calc)
        print(result)
        print()
    

def demo_detailed_retrieval():
    """Show retrieval statistics"""
    print("="*60)
    print("DETAILED RETRIEVAL ANALYSIS")
    print("="*60)

    rag_tool = NutritionRAGTool()
    query = "What are macronutrients?"
    print(f"\n Query:  {query}\n")

    # Get detailed stats
    stats = rag_tool.get_retrieval_stats(query)
    print(" Retrieval statistics:")
    print(f"    Response lenght: {stats['response_length']} characters")
    print(f"    Number of sources: {stats['num_sources']}")

    if 'source_details' in stats:
        print("\n Source Details :")
        for i, source in enumerate(stats['source_details'], 1):
            print(f"\n  {i}. {source['book']} (Page {source['page']})")
            print(f"    Relevance Score: {source['score']:.4f}")
            print(f"    Chunk Length: {source['text_length']} chars")

def interactive_mode():
    """Interactive query mode"""

    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Ask nutrition quetion! (Type 'quit' to exit)")
    print()


    rag_tool = NutritionRAGTool()

    while True:
        query= input("Your question: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("\n Goodbye")
            break

        if not query:
            continue

        print("Searcning ....\n")
        result = rag_tool._run(query)

        print("Answer:")
        print("-"*60)
        print(result)
        print("-"*60)

def main():
    """Main demo function"""

    print("\n" + "="*60)
    print(" RAG TOOL DEMO MENU")
    print("\n1. Sample Queries Demo")
    print("2. Calculator Tool Demo")
    print("3. Detailed Retrieval Analysis")
    print("4. Interactive mode")
    print("5. Run All")
    print()

    choice = input("Select option (1-5)".strip)

    if choice == "1":
        demo_rag_tool()
    elif choice == "2":
        demo_calculator_tool()
    elif choice == "3":
        demo_detailed_retrieval()
    elif choice == "4":
        interactive_mode()
    elif choice =="5":
        demo_rag_tool()
        demo_calculator_tool()
        demo_detailed_retrieval()
    else:
        print("Invalid option")
        return
    
    print("\n" + "="*60)
    print("Demo complete")
    print("="*60)


if __name__ == "__main__":
    main()

