import chromadb
from sentence_transformers import SentenceTransformer
from ollama import chat

from src.config import TOP_K, MAX_DISTANCE, OLLAMA_MODEL


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path="vectorstore"
)

collection = client.get_collection(
    name="documents"
)


def answer_question(question):
    question_embedding = model.encode([question])

    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=TOP_K
    )

    retrieved_chunks = results["documents"][0]
    distances = results["distances"][0]
    metadata = results["metadatas"][0]

    filtered_chunks = []
    filtered_sources = []

    for chunk, distance, source in zip(
        retrieved_chunks,
        distances,
        metadata
    ):
        if distance <= MAX_DISTANCE:
            filtered_chunks.append(chunk)
            filtered_sources.append(source)

    retrieved_chunks = filtered_chunks
    sources = filtered_sources
    if not retrieved_chunks:
        return "I don't know based on the provided documents.", []

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using only the information provided in the context.

Give a clear, complete answer in one or two sentences.
Do not add information that is not present in the context.
If the answer cannot be found in the context, say:
"I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

    response = chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.message.content, sources

def get_chunk_count():
    return collection.count()