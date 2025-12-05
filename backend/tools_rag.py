# backend/tools_rag.py

import os
from dotenv import load_dotenv

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.tools import tool

load_dotenv()

# Path to your vector DB
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")

# Load embedding model (same one used in vector DB build)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path=DB_DIR)

# Load the collection
collection = chroma_client.get_or_create_collection(
    name="course_rag",
    metadata={"hnsw:space": "cosine"},
)


# ------------------------------
# RAG TOOL: course_docs_search
# ------------------------------
@tool("course_docs_search", return_direct=False)
def course_docs_search(query: str) -> str:
    """
    Search the Machine Learning course documents (vector DB)
    and return the most relevant passages.
    """
    # Embed query
    query_embedding = embedder.encode([query]).tolist()[0]

    # Retrieve top 5 chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    documents = results.get("documents", [[]])[0]
    sources = results.get("metadatas", [[]])[0]

    response = "📚 **Relevant excerpts from course materials:**\n\n"
    for i, doc in enumerate(documents):
        src = sources[i].get("source", "unknown")
        response += f"**From {src}:**\n{doc}\n\n---\n\n"

    return response
