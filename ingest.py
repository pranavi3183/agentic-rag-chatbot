from app.ingestion import load_and_split_pdf
from app.vector_store import create_vector_store

PDF_PATH = "data/Ebook-Agentic-AI.pdf"


def main():
    print("Loading PDF...")

    chunks = load_and_split_pdf(PDF_PATH)

    print(f"Created {len(chunks)} chunks.")

    print("Uploading chunks to Pinecone...")

    create_vector_store(chunks)

    print("Ingestion completed successfully!")


if __name__ == "__main__":
    main()