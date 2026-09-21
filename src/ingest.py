import shutil
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.chunker import chunk_text
from src.document_loader import load_document


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


vectorstore_path = "vectorstore"

shutil.rmtree(vectorstore_path, ignore_errors=True)

client = chromadb.PersistentClient(
    path=vectorstore_path
)

collection = client.get_or_create_collection(
    name="documents"
)


documents = []

for pattern in ["*.txt", "*.md", "*.pdf"]:
    documents.extend(Path("docs").glob(pattern))


print("Documents found:")

for document in documents:
    print(document)


all_chunks = []
all_metadata = []
all_ids = []

chunk_id = 0


for document_path in documents:

    text = load_document(document_path)

    chunks = chunk_text(text)

    for chunk in chunks:

        all_chunks.append(chunk)

        all_metadata.append({
            "source": str(document_path)
        })

        all_ids.append(f"chunk_{chunk_id}")

        chunk_id += 1


embeddings = model.encode(all_chunks)


collection.add(
    ids=all_ids,
    documents=all_chunks,
    embeddings=embeddings.tolist(),
    metadatas=all_metadata
)


print()
print("Documents successfully added to Chroma!")

print()
print(f"Number of chunks stored: {collection.count()}")

print()
print("All stored chunks:")


stored_chunks = collection.get()


for i, chunk in enumerate(stored_chunks["documents"]):

    print()
    print(f"--- Chunk {i} ---")
    print(chunk)
    print(f"Source: {stored_chunks['metadatas'][i]['source']}")