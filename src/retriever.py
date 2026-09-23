import chromadb
from sentence_transformers import SentenceTransformer

from src.config import TOP_K, MAX_DISTANCE
from src.bm25_index import build_bm25_retriever
from src.reranker import rerank


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path="vectorstore"
)

collection = client.get_collection(
    name="documents"
)


def semantic_retrieve(question, top_k=TOP_K):
    question_embedding = model.encode([question])

    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=top_k
    )

    retrieved_chunks = results["documents"][0]
    distances = results["distances"][0]
    metadata = results["metadatas"][0]

    results_list = []

    for chunk, distance, source in zip(
        retrieved_chunks,
        distances,
        metadata
    ):
        if distance <= MAX_DISTANCE:
            results_list.append(
                {
                    "chunk": chunk,
                    "score": float(distance),
                    "source": source
                }
            )

    return results_list


bm25_retriever = build_bm25_retriever()


def keyword_retrieve(question, top_k=TOP_K):
    return bm25_retriever.retrieve(
        question,
        top_k=top_k
    )


def hybrid_retrieve(
    question,
    top_k=TOP_K,
    use_reranker=True
):
    # Retrieve more candidates than the final number of chunks.
    candidate_k = max(top_k * 2, 5)

    semantic_results = semantic_retrieve(
        question,
        top_k=candidate_k
    )

    keyword_results = keyword_retrieve(
        question,
        top_k=candidate_k
    )

    scores = {}
    chunks = {}

    # Semantic results
    for rank, result in enumerate(
        semantic_results,
        start=1
    ):
        chunk = result["chunk"]

        chunks[chunk] = {
            "chunk": chunk,
            "source": result["source"]
        }

        scores[chunk] = scores.get(chunk, 0) + (
            1 / (60 + rank)
        )

    # BM25 results
    for rank, result in enumerate(
        keyword_results,
        start=1
    ):
        chunk = result["chunk"]

        if chunk not in chunks:
            chunks[chunk] = {
                "chunk": chunk,
                "source": result["source"]
            }

        scores[chunk] = scores.get(chunk, 0) + (
            1 / (60 + rank)
        )

    ranked_chunks = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    hybrid_results = [
        {
            "chunk": chunks[chunk]["chunk"],
            "score": scores[chunk],
            "source": chunks[chunk]["source"]
        }
        for chunk in ranked_chunks
    ]

    if not use_reranker:
        return hybrid_results[:top_k]

    # Rerank the hybrid candidate set.
    return rerank(
        question,
        hybrid_results,
        top_k=top_k
    )