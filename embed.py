"""
embed.py

Takes the job documents from load_data.py, splits them into chunks,
embeds each chunk using a free local model, and saves the FAISS
vector index to disk.

Run this once to build the index. After that, the app loads it
directly without re-embedding.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from load_data import load_job_documents

INDEX_PATH = "joblens_index"


def build_index():
    # Step 1: Load raw job documents
    print("Loading job descriptions...")
    raw_docs = load_job_documents()

    # Step 2: Split into chunks
    # chunk_size=500 = ~500 characters per chunk
    # chunk_overlap=50 = last 50 chars of one chunk repeat at the start
    # of the next, so we don't lose context at cut points
    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    # Wrap in LangChain Document objects (needed for FAISS)
    documents = []
    for text in raw_docs:
        chunks = splitter.split_text(text)
        for chunk in chunks:
            documents.append(Document(page_content=chunk))

    print(f"Created {len(documents)} chunks from {len(raw_docs)} job descriptions.")

    # Step 3: Load the embedding model
    # all-MiniLM-L6-v2 is a small, fast, free model that runs locally.
    # First run downloads it (~90MB). After that it's cached.
    print("Loading embedding model (downloads ~90MB on first run)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Step 4: Embed all chunks and store in FAISS
    print("Embedding chunks and building FAISS index...")
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Step 5: Save to disk so we don't have to re-embed every time
    vectorstore.save_local(INDEX_PATH)
    print(f"Index saved to '{INDEX_PATH}'. Done!")


if __name__ == "__main__":
    build_index()
