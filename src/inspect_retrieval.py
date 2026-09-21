import chromadb
from sentence_transformers import SentenceTransformer


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path="vectorstore"
)

collection = client.get_collection(
    name="documents"
)


question = "Which team is responsible for getting new customers?"
question_embedding = model.encode([question])


results = collection.query(
    query_embeddings=question_embedding.tolist(),
    n_results=3
)


print("Question:")
print(question)

print()

for i, document in enumerate(results["documents"][0]):

    print(f"--- Result {i + 1} ---")
    print("Distance:", results["distances"][0][i])
    print("Source:", results["metadatas"][0][i]["source"])
    print("Text:", document)
    print()