import chromadb

from src.bm25_retriever import BM25Retriever


client = chromadb.PersistentClient(
    path="vectorstore"
)

collection = client.get_collection(
    name="documents"
)


def build_bm25_retriever():
    results = collection.get(
        include=["documents", "metadatas"]
    )

    documents = [
        {
            "chunk": chunk,
            "source": metadata
        }
        for chunk, metadata in zip(
            results["documents"],
            results["metadatas"]
        )
    ]

    return BM25Retriever(documents)