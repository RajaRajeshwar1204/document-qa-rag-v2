from src.retriever import (
    semantic_retrieve,
    keyword_retrieve,
    hybrid_retrieve,
)


questions = [
    {
        "question": "Where is Acme Corporation headquartered?",
        "expected_keyword": "Hyderabad",
    },
    {
        "question": "When was Acme Corporation founded?",
        "expected_keyword": "2015",
    },
    {
        "question": "When was Acme Cloud launched?",
        "expected_keyword": "2021",
    },
    {
        "question": "What departments does Acme Corporation have?",
        "expected_keyword": "Engineering",
    },
    {
        "question": "Which team is responsible for getting new customers?",
        "expected_keyword": "Sales",
    },
]


def evaluate(results, expected_keyword):
    return any(
        expected_keyword.lower() in result["chunk"].lower()
        for result in results
    )


methods = {
    "Semantic": lambda question: semantic_retrieve(
        question,
        top_k=2,
    ),
    "BM25": lambda question: keyword_retrieve(
        question,
        top_k=2,
    ),
    "Hybrid RRF": lambda question: hybrid_retrieve(
        question,
        top_k=2,
        use_reranker=False,
    ),
    "Hybrid + Reranker": lambda question: hybrid_retrieve(
        question,
        top_k=2,
        use_reranker=True,
    ),
}


scores = {method: 0 for method in methods}


for item in questions:
    question = item["question"]
    expected_keyword = item["expected_keyword"]

    print()
    print("=" * 70)
    print("Question:", question)
    print("Expected keyword:", expected_keyword)

    for method_name, retrieve in methods.items():
        results = retrieve(question)

        passed = evaluate(
            results,
            expected_keyword,
        )

        if passed:
            scores[method_name] += 1

        print()
        print(method_name + ":")
        print("  Retrieved:", "YES" if passed else "NO")

        for result in results:
            print(
                "  -",
                result["source"],
                "| score:",
                result["score"],
            )


print()
print("=" * 70)
print("RETRIEVAL COMPARISON")
print("=" * 70)

for method_name, score in scores.items():
    recall = score / len(questions)

    print(
        f"{method_name}: "
        f"{score}/{len(questions)} "
        f"({recall:.0%})"
    )

print("=" * 70)