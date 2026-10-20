
from app.services.retrieval_service import search_documents


# These expected pages are based on our own sample PDF.
test_cases = [
    {
        "question": "Which documents are required for a claim?",
        "expected_page": 2,
    },
    {
        "question": "How are premiums paid?",
        "expected_page": 3,
    },
    {
        "question": "What does the policy say about exclusions?",
        "expected_page": 3,
    },
    {
        "question": "How can I renew my insurance?",
        "expected_page": 3,
    },
    {
        "question": "What does the policy say about coverage limits?",
        "expected_page": 1,
    },
]

hits = 0
reciprocal_ranks = []

print("\nRETRIEVAL EVALUATION")
print("-" * 70)

for case in test_cases:
    results = search_documents(case["question"], top_k=3)

    relevant_rank = next(
        (
            rank
            for rank, result in enumerate(results, start=1)
            if result["page"] == case["expected_page"]
        ),
        None,
    )

    hit = relevant_rank is not None
    hits += int(hit)

    reciprocal_rank = (
        1 / relevant_rank if relevant_rank is not None else 0
    )
    reciprocal_ranks.append(reciprocal_rank)

    print("\nQuestion:", case["question"])
    print("Expected page:", case["expected_page"])
    print("Retrieved pages:", [r["page"] for r in results])
    print("Hit@3:", hit)
    print("Reciprocal rank:", round(reciprocal_rank, 3))


recall_at_3 = hits / len(test_cases)
mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

print("\n" + "-" * 70)
print(f"Recall@3: {recall_at_3:.3f}")
print(f"MRR:      {mrr:.3f}")
