import chromadb
from sentence_transformers import SentenceTransformer
from ollama import chat


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

retrieved_chunks = results["documents"][0]

context = "\n\n".join(retrieved_chunks)

prompt = f"""
Answer the question using only the information in the context below.

Context:
{context}

Question:
{question}

Answer:
"""

response = chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("Answer:")
print(response.message.content)

print()
print("Sources:")

for i, chunk in enumerate(retrieved_chunks):
    print(f"--- Source {i + 1} ---")
    print(chunk)
    print()