def route_question(question):
    question_lower = question.lower()

    document_keywords = [
        "acme",
        "company",
        "product",
        "cloud",
        "department",
        "team",
        "customer",
        "customers",
        "document",
        "workflow",
        "headquarters",
        "founded",
        "launched",
        "employee",
        "revenue"
    ]

    if any(keyword in question_lower for keyword in document_keywords):
        return "DOCUMENT"

    return "OTHER"


if __name__ == "__main__":
    print(route_question("When was Acme Cloud launched?"))
    print(route_question("Who is the CEO of Acme Corporation?"))
    print(route_question("Which team is responsible for getting new customers?"))
    print(route_question("When was Acme Corporation founded on Mars?"))
    print(route_question("Hello, how are you?"))