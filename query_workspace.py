# query_workspace.py
import sys

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

QDRANT_URL = "http://localhost:6333"


def query(collection_name: str, query_text: str, k: int = 5):
    """Query the vector store and return similar documents."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=collection_name,
    )

    results = vectorstore.similarity_search(query_text, k=k)

    for i, doc in enumerate(results, 1):
        print(f"\n{'='*60}")
        print(f"Result {i}: {doc.metadata.get('source', 'unknown')}")
        print(f"{'='*60}")
        print(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)

    return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python query_workspace.py <collection_name> <query>")
        print("Example: python query_workspace.py workspace_H2BWebApps 'authentication logic'")
        sys.exit(1)

    collection = sys.argv[1]
    query_text = " ".join(sys.argv[2:])
    query(collection, query_text)
