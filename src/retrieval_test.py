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

question = "Where is Acme Corporation headquartered?"

question_embedding = model.encode([question])

results = collection.query(
    query_embeddings=question_embedding.tolist(),
    n_results=2
)

print("Full retrieval result:")
print(results)
print()

for i, document in enumerate(results["documents"][0]):
    print(f"--- Result {i + 1} ---")
    print(document)
    print()