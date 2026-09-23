from sentence_transformers import CrossEncoder


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


model = CrossEncoder(RERANKER_MODEL)


def rerank(question, results, top_k=3):
    if not results:
        return []

    pairs = [
        (question, result["chunk"])
        for result in results
    ]

    scores = model.predict(pairs)

    reranked = []

    for result, score in zip(results, scores):
        reranked.append(
            {
                "chunk": result["chunk"],
                "source": result["source"],
                "score": float(score)
            }
        )

    reranked.sort(
        key=lambda result: result["score"],
        reverse=True
    )

    return reranked[:top_k]