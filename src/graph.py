from typing import TypedDict

from langgraph.graph import StateGraph, END

from src.retriever import hybrid_retrieve
from src.config import TOP_K, OLLAMA_MODEL
from src.router import route_question

from ollama import chat


class RAGState(TypedDict):
    question: str
    context: list[str]
    sources: list[str]
    distances: list[float]
    retrieval_status: str
    answer: str
    verification: str
    retries: int
    verification_attempts: int
    route: str


def route(state: RAGState):
    decision = route_question(state["question"])

    return {
        "route": decision
    }


def retrieve(state: RAGState):
    results = hybrid_retrieve(
        state["question"],
        top_k=TOP_K
    )

    return {
        "context": [
            result["chunk"]
            for result in results
        ],
        "sources": [
            result["source"]["source"]
            for result in results
        ],
        "distances": [
            result["score"]
            for result in results
        ]
    }


def check_retrieval(state: RAGState):
    if not state["context"]:
        print("Retrieval: NOT_RELEVANT")

        return {
            "retrieval_status": "NOT_RELEVANT"
        }

    print("Retrieval: RELEVANT")

    return {
        "retrieval_status": "RELEVANT"
    }


def generate_answer(state: RAGState):
    context = "\n\n".join(state["context"])

    prompt = f"""
Answer the question using only the provided context.

If the context contains the answer, answer the question directly.
If the question asks for multiple items, include all relevant items
that are explicitly stated in the context.

Only say:
"I don't know based on the provided documents."
when the requested information is genuinely absent from the context.

Context:
{context}

Question:
{state["question"]}

Answer:
"""

    response = chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0
        }
    )

    return {
        "answer": response.message.content
    }


def verify_answer(state: RAGState):
    context = "\n\n".join(state["context"])

    attempt = state["verification_attempts"] + 1

    prompt = f"""
You are a strict fact checker.

Decide whether the answer is supported by the context.

Read the answer carefully.

If the answer is consistent with and supported by information in the context,
reply exactly:
SUPPORTED

If the answer contains claims that are not supported by the context,
reply exactly:
NOT_SUPPORTED

Do not require the answer to use the exact same wording as the context.
Paraphrases and additional explanatory wording are allowed as long as
the factual claims remain supported by the context.

Context:
{context}

Answer:
{state["answer"]}

Verdict:
"""

    response = chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0
        }
    )

    verdict_text = response.message.content.strip().upper()

    if verdict_text == "NOT_SUPPORTED":
        verdict = "NOT_SUPPORTED"
    else:
        verdict = "SUPPORTED"

    print(f"Verifier: {verdict}")

    return {
        "answer": state["answer"],
        "verification": verdict,
        "verification_attempts": attempt
    }


def handle_failed_verification(state: RAGState):
    return {
        "answer": "I don't know based on the provided documents.",
        "verification": "NOT_SUPPORTED",
        "retries": state["retries"] + 1,
        "verification_attempts": state["verification_attempts"]
    }


def retry_answer(state: RAGState):
    context = "\n\n".join(state["context"])

    prompt = f"""
Answer the question again using only the provided context.

Be precise and do not add information that is not present in the context.

If the context contains the answer, answer the question directly.
If the question asks for multiple items, include all relevant items
that are explicitly stated in the context.

Only say:
"I don't know based on the provided documents."
when the requested information is genuinely absent from the context.

Context:
{context}

Question:
{state["question"]}

Previous answer:
{state["answer"]}

Answer:
"""

    response = chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0
        }
    )

    return {
        "answer": response.message.content,
        "verification": "",
        "retries": state["retries"] + 1,
        "verification_attempts": state["verification_attempts"]
    }


def handle_other(state: RAGState):
    return {
        "answer": "I don't know based on the provided documents.",
        "verification": "NOT_APPLICABLE",
        "sources": []
    }


def handle_irrelevant_retrieval(state: RAGState):
    return {
        "answer": "I don't know based on the provided documents.",
        "verification": "NOT_APPLICABLE"
    }


def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("route", route)
    graph.add_node("retrieve", retrieve)
    graph.add_node("check_retrieval", check_retrieval)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("verify", verify_answer)
    graph.add_node("failed_verification", handle_failed_verification)
    graph.add_node("retry_answer", retry_answer)
    graph.add_node("other", handle_other)
    graph.add_node(
        "irrelevant_retrieval",
        handle_irrelevant_retrieval
    )

    graph.set_entry_point("route")

    graph.add_conditional_edges(
        "route",
        lambda state: state["route"],
        {
            "DOCUMENT": "retrieve",
            "OTHER": "other"
        }
    )

    graph.add_edge("other", END)

    graph.add_edge("retrieve", "check_retrieval")

    graph.add_conditional_edges(
        "check_retrieval",
        lambda state: state["retrieval_status"],
        {
            "RELEVANT": "generate_answer",
            "NOT_RELEVANT": "irrelevant_retrieval"
        }
    )

    graph.add_edge("irrelevant_retrieval", END)

    graph.add_edge("generate_answer", "verify")

    graph.add_conditional_edges(
        "verify",
        lambda state: (
            "end"
            if state["verification"] == "SUPPORTED"
            else "retry"
            if state["retries"] < 1
            else "failed"
        ),
        {
            "end": END,
            "retry": "retry_answer",
            "failed": "failed_verification"
        }
    )

    graph.add_edge("retry_answer", "verify")
    graph.add_edge("failed_verification", END)

    return graph.compile()


def ask_question(question):
    graph = build_graph()

    result = graph.invoke({
        "question": question,
        "context": [],
        "sources": [],
        "distances": [],
        "retrieval_status": "",
        "answer": "",
        "verification": "",
        "retries": 0,
        "verification_attempts": 0,
        "route": ""
    })

    return (
        result["answer"],
        result["verification"],
        result["sources"],
        result["verification_attempts"]
    )


if __name__ == "__main__":
    answer, verification, sources, verification_attempts = ask_question(
        "When was Acme Cloud launched?"
    )

    print()
    print("Answer:", answer)
    print("Verification:", verification)
    print("Verification attempts:", verification_attempts)
    print("Sources:", sources)