import re

from rank_bm25 import BM25Okapi


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents

        tokenized_documents = [
            tokenize(document["chunk"])
            for document in documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    def retrieve(self, query, top_k=5):
        tokenized_query = tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        return [
            {
                "chunk": self.documents[i]["chunk"],
                "score": float(scores[i]),
                "source": self.documents[i]["source"]
            }
            for i in ranked_indices
        ]