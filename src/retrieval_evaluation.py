import chromadb
from sentence_transformers import SentenceTransformer


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path="vectorstore")
collection = client.get_collection(name="documents")


questions = [
    {
        "question": "Where is Acme Corporation headquartered?",
        "expected_keyword": "Hyderabad"
    },
    {
        "question": "When was Acme Corporation founded?",
        "expected_keyword": "2015"
    },
    {
        "question": "When was Acme Cloud launched?",
        "expected_keyword": "2021"
    },
    {
        "question": "What departments does Acme Corporation have?",
        "expected_keyword": "Engineering"
    },
    {
        "question": "Which team is responsible for getting new customers?",
        "expected_keyword": "Sales"
    },
]

correct = 0

for item in questions:
    question = item["question"]
    expected_keyword = item["expected_keyword"]

    embedding = model.encode([question])

    results = collection.query(
        query_embeddings=embedding.tolist(),
        n_results=2
    )

    retrieved_chunks = results["documents"][0]

    found = any(
        expected_keyword.lower() in chunk.lower()
        for chunk in retrieved_chunks
    )

    if found:
        correct += 1

    print()
    print("=" * 60)
    print("Question:", question)
    print("Expected keyword:", expected_keyword)
    print("Retrieved:", "YES" if found else "NO")


print()
print("=" * 60)

recall = correct / len(questions)

print(f"Retrieval Recall@2: {correct}/{len(questions)}")
print(f"Retrieval Recall@2: {recall:.0%}")

print("=" * 60)