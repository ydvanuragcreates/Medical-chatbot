"""Build the Pinecone index from the PDFs in Data/.

Usage:  python store_index.py
The index name comes from PINECONE_INDEX (default "medicalbot"). If the index
already has vectors, the script stops rather than adding duplicates; pass
--reset to wipe it and rebuild.
"""

import sys

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.config import get_settings
from src.helper import download_hugging_face_embeddings, load_pdf_file, text_split

settings = get_settings()
if not settings.pinecone_api_key:
    sys.exit("PINECONE_API_KEY is not set (check your .env).")

reset = "--reset" in sys.argv
index_name = settings.pinecone_index

print("Loading PDFs from Data/ ...")
pages = load_pdf_file(data="Data/")
chunks = text_split(pages)
print(f"{len(pages)} pages -> {len(chunks)} chunks")

pc = Pinecone(api_key=settings.pinecone_api_key)
if not pc.has_index(index_name):
    print(f"Creating index '{index_name}'")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)
existing = index.describe_index_stats().get("total_vector_count", 0)
if existing:
    if not reset:
        sys.exit(
            f"Index '{index_name}' already holds {existing} vectors. "
            "Re-run with --reset to wipe and rebuild it, or set PINECONE_INDEX "
            "to a new index name."
        )
    print(f"Deleting {existing} existing vectors from '{index_name}'")
    index.delete(delete_all=True)

print(f"Embedding and uploading to '{index_name}' (this takes a few minutes) ...")
PineconeVectorStore.from_documents(
    documents=chunks,
    index_name=index_name,
    embedding=download_hugging_face_embeddings(settings.embedding_model),
    batch_size=64,
)
print("Done.")
