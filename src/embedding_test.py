from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


sentences = [
    "The cat is sleeping.",
    "The kitten is taking a nap.",
    "The stock market increased today."
]


embeddings = model.encode(sentences)


similarity_1_2 = cos_sim(embeddings[0], embeddings[1])

similarity_1_3 = cos_sim(embeddings[0], embeddings[2])


print("Similarity between sentence 1 and sentence 2:")
print(similarity_1_2)

print()

print("Similarity between sentence 1 and sentence 3:")
print(similarity_1_3)