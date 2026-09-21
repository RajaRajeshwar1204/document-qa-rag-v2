from document_loader import load_text_file


text = load_text_file("docs/company.txt")

print("Type of document:")
print(type(text))

print()

print("Number of characters:")
print(len(text))

print()

print("First 100 characters:")
print(text[:100])