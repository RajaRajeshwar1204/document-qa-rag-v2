from pathlib import Path
from pypdf import PdfReader


def load_text_file(file_path):
    """
    Read a text file and return its contents.
    """
    path = Path(file_path)

    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def load_pdf_file(file_path):
    """
    Read a PDF file and return its text.
    """
    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text


def load_document(file_path):
    """
    Load a document based on its file type.
    """
    path = Path(file_path)

    if path.suffix.lower() == ".pdf":
        return load_pdf_file(path)

    return load_text_file(path)