from ollama import chat

from src.config import OLLAMA_MODEL
from src.retriever import hybrid_retrieve


def answer_question(question):
    retrieval_results = hybrid_retrieve(question)

    retrieved_chunks = [
        result["chunk"]
        for result in retrieval_results
    ]

    sources = [
        result["source"]
        for result in retrieval_results
    ]

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
    return 0