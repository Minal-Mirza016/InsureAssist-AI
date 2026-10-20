
from app.services.retrieval_service import search_documents
from app.services.llm_service import generate_answer


question = "Which documents are required for a claim?"

# Step 1: Retrieve relevant passages from the policy PDF.
results = search_documents(question, top_k=3)

# Step 2: Format the retrieved passages with source details.
context_parts = []

for result in results:
    context_parts.append(
        f"Source: {result['source']}, "
        f"Page: {result['page']}\n"
        f"Passage: {result['text']}"
    )

context = "\n\n".join(context_parts)

# Step 3: Ask the local LLM to answer using that context.
answer = generate_answer(question, context)

print("\nQUESTION")
print(question)

print("\nRETRIEVED CONTEXT")
print(context)

print("\nGENERATED ANSWER")
print(answer)
