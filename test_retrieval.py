
from app.services.retrieval_service import search_documents


questions = [
    "Which documents do I need to submit a claim?",
    "How are premium payments handled?",
    "What does the sample policy say about exclusions?",
    "How can I renew my policy?",
]


for question in questions:
    print("\n" + "=" * 80)
    print("QUESTION:", question)

    results = search_documents(question, top_k=3)

    for rank, result in enumerate(results, start=1):
        print(f"\nResult {rank}")
        print("Source:", result["source"])
        print("Page:", result["page"])
        print("Similarity:", result["similarity"])
        print("Passage:", result["text"])
