#!/usr/bin/env python3
import logging
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from vector_store_config import get_qdrant_client, get_embeddings

logger = logging.getLogger(__name__)

# Cache for workspace root paths (collection_name -> root_path)
_workspace_roots: dict[str, str] = {}

# Use dependency injection for clients
embeddings = get_embeddings()
qdrant_client = get_qdrant_client()


def _to_workspace_id(workspace_id: str) -> str:
    """Creates a workspace id out of a given unique workspace name for unique index"""
    if "/" in workspace_id:
        workspace_id_name = workspace_id.split("/")[-1]
    else:
        workspace_id_name = workspace_id

    return f"workspace_{workspace_id_name}"


def get_workspace_root(workspace: str) -> str:
    """Get the workspace root path for a collection. No worries this is cached."""
    collection_name = _to_workspace_id(workspace)
    if collection_name in _workspace_roots:
        logger.debug(f"_workspace_roots found in cache {_workspace_roots}")
        return _workspace_roots[collection_name]

    # Try to get from collection metadata or first document first time only
    try:
        vectorstore = QdrantVectorStore(
            embedding=embeddings,
            client=qdrant_client,
            collection_name=collection_name,
        )
        results = vectorstore.similarity_search("", k=1)
        if results:
            source = results[0].metadata.get("source", "")
            # Find common prefix by looking for workspace pattern
            # Assumes paths like /path/to/workspace/src/file.ts
            import os

            parts = source.split(os.sep)
            # Look for common workspace indicators
            for i, part in enumerate(parts):
                if part in ("src", "apps", "libs", "packages", "components"):
                    root = os.sep.join(parts[:i])
                    _workspace_roots[collection_name] = root
                    return root
    except Exception:
        pass
    return ""


def search_with_filter(*, workspace: str, query: str, filter: str, k: int):
    """Search with layer filter using Qdrant native filtering."""
    vectorstore = store_for_workspace(workspace)
    filter_condition = Filter(
        must=[FieldCondition(key="metadata.layer", match=MatchValue(value=filter))]
    )
    return vectorstore.similarity_search(query, k=k, filter=filter_condition)


def store_for_workspace(workspace: str) -> QdrantVectorStore:
    collection_name = _to_workspace_id(workspace)
    vectorstore = QdrantVectorStore(
        collection_name=collection_name,
        embedding=embeddings,
        client=qdrant_client,
    )
    return vectorstore


__all__ = ["store_for_workspace", "get_workspace_root", "search_with_filter"]
