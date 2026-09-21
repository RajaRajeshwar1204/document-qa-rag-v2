from src.graph import ask_question


questions = [
    {
        "question": "Where is Acme Corporation headquartered?",
        "expected_answer": "Hyderabad",
        "should_refuse": False
    },
    {
        "question": "When was Acme Corporation founded?",
        "expected_answer": "2015",
        "should_refuse": False
    },
    {
        "question": "When was Acme Cloud launched?",
        "expected_answer": "2021",
        "should_refuse": False
    },
    {
        "question": "What departments does Acme Corporation have?",
        "expected_answer": "Engineering, Sales, and Customer Success",
        "should_refuse": False
    },
    {
        "question": "Which team is responsible for getting new customers?",
        "expected_answer": "Sales",
        "should_refuse": False
    },
    {
        "question": "What is Acme Corporation's annual revenue?",
        "expected_answer": "I don't know based on the provided documents.",
        "should_refuse": True
    },
    {
        "question": "What does Acme Cloud help businesses do?",
        "expected_keywords": ["documents", "workflows"],
        "should_refuse": False
    },
    {
        "question": "Who is the CEO of Acme Corporation?",
        "expected_answer": "I don't know based on the provided documents.",
        "should_refuse": True
    },
    {
        "question": "How many employees does Acme Corporation have?",
        "expected_answer": "I don't know based on the provided documents.",
        "should_refuse": True
    },
    {
        "question": "What programming language does Acme Cloud use?",
        "expected_answer": "I don't know based on the provided documents.",
        "should_refuse": True
    },
    {
        "question": "When was Acme Corporation founded on Mars?",
        "expected_answer": "I don't know based on the provided documents.",
        "should_refuse": True
    },
]


correct = 0


for item in questions:
    question = item["question"]

    answer, verification, sources, verification_attempts = ask_question(
        question
    )

    print()
    print("=" * 60)
    print("Question:", question)

    if "expected_answer" in item:
        print("Expected:", item["expected_answer"])
    else:
        print("Expected keywords:", item["expected_keywords"])

    print("Actual:", answer)
    print("Verification:", verification)
    print("Verification attempts:", verification_attempts)

    print("Sources:")
    if sources:
        for source in sources:
            print(" -", source)
    else:
        print(" - None")

    passed = False

    if item["should_refuse"]:
        if "I don't know based on the provided documents." in answer:
            passed = True

    elif "expected_keywords" in item:
        answer_lower = answer.lower()

        if all(
            keyword.lower() in answer_lower
            for keyword in item["expected_keywords"]
        ):
            passed = True

    else:
        if item["expected_answer"].lower() in answer.lower():
            passed = True

    if passed:
        correct += 1
        print("Result: PASS")
    else:
        print("Result: FAIL")


print()
print("=" * 60)
print(f"Score: {correct}/{len(questions)}")
print("=" * 60)