# query_workspace.py
import sys
import argparse

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Filter, FieldCondition, MatchValue

from config import QDRANT_URL, EMBEDDING_MODEL


def query(collection_name: str, query_text: str, k: int = 5, layer: str = None):
    """Query the vector store and return similar documents."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=collection_name,
    )

    # Build filter if layer specified
    filter_condition = None
    if layer:
        filter_condition = Filter(
            must=[FieldCondition(key="metadata.layer", match=MatchValue(value=layer))]
        )

    results = vectorstore.similarity_search(query_text, k=k, filter=filter_condition)

    for i, doc in enumerate(results, 1):
        layer_tag = f" [{doc.metadata.get('layer', '')}]" if doc.metadata.get('layer') else ""
        print(f"\n{'='*60}")
        print(f"Result {i}: {doc.metadata.get('source', 'unknown')}{layer_tag}")
        print(f"{'='*60}")
        print(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query a workspace collection")
    parser.add_argument("collection", help="Collection name (e.g., workspace_H2BWebApps)")
    parser.add_argument("query", nargs="+", help="Search query")
    parser.add_argument("-k", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--layer", "-l", help="Filter by layer (architecture, services, components, utils)")

    args = parser.parse_args()

    query_text = " ".join(args.query)
    query(args.collection, query_text, k=args.k, layer=args.layer)
