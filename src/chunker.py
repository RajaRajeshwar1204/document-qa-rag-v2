def chunk_text(text, sentences_per_chunk=4):
    """
    Group sentences into chunks while keeping related information together.
    """

    sentences = [
        sentence.strip()
        for sentence in text.replace("\n\n", " ").split(".")
        if sentence.strip()
    ]

    chunks = []

    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = ". ".join(sentences[i:i + sentences_per_chunk]) + "."
        chunks.append(chunk)

    return chunks


if __name__ == "__main__":
    from document_loader import load_text_file

    text = load_text_file("docs/company.txt")
    chunks = chunk_text(text)

    print(f"Number of chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        print()
        print(f"--- Chunk {i + 1} ---")
        print(chunk)