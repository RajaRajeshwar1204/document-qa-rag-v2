from src.graph import ask_question


questions = [
    "Where is Acme Corporation headquartered?",
    "When was Acme Corporation founded?",
    "When was Acme Cloud launched?",
    "What departments does Acme Corporation have?",
    "Which team is responsible for getting new customers?",
    "What does Acme Cloud help businesses do?",
    "Who is the CEO of Acme Corporation?",
    "How many employees does Acme Corporation have?",
    "What programming language does Acme Cloud use?",
    "When was Acme Corporation founded on Mars?"
]


for question in questions:
    print()
    print("=" * 70)
    print("Question:", question)

    answer, verification, sources, attempts = ask_question(question)

    print("Answer:", answer)
    print("Verification:", verification)
    print("Attempts:", attempts)
    print("Sources:")

    for source in sources:
        print(" -", source)