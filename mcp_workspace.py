#!/usr/bin/env python3
import asyncio
from mcp.server.fastmcp import FastMCP
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from config import QDRANT_URL, EMBEDDING_MODEL
from layer_config import (
    classify_layer,
    load_default_layer_map,
    get_architecture_layers,
    get_base_layers,
    load_project_layer_map,
)
from index_workspace import load_workspace, split_documents, create_vectorstore

mcp = FastMCP("workspace-genie")

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
qdrant_client = QdrantClient(url=QDRANT_URL)


def _format_results(results) -> list:
    """Format search results into structured list."""
    return [
        {
            "file": r.metadata.get("source", "unknown"),
            "layer": r.metadata.get("layer", "other"),
            "content": r.page_content[:500],
        }
        for r in results
    ]


def _format_section(items: list) -> str:
    """Format a section of results as markdown."""
    if not items:
        return "_No relevant items found_\n"

    output = ""
    for item in items:
        output += f"\n### {item['file']}\n```\n{item['content']}\n```\n"
    return output


def _search_with_filter(vectorstore, query: str, layer: str, k: int):
    """Search with layer filter using Qdrant native filtering."""
    filter_condition = Filter(
        must=[FieldCondition(key="metadata.layer", match=MatchValue(value=layer))]
    )
    return vectorstore.similarity_search(query, k=k, filter=filter_condition)


def _search_with_layers(vectorstore, query: str, layers: list[str], k: int):
    """Search across multiple layers, distributing k across them."""
    if not layers:
        return []

    results = []
    k_per_layer = max(1, k // len(layers))

    for layer in layers:
        try:
            layer_results = _search_with_filter(vectorstore, query, layer, k_per_layer)
            results.extend(layer_results)
        except Exception:
            continue

    return results[:k]


@mcp.tool()
def search_codebase(workspace: str, query: str, num_results: int = 5) -> str:
    """Search indexed workspace for relevant code/files.

    Args:
        workspace: Name of the workspace collection (e.g., "YourAppName")
        query: Search query to find relevant code
        num_results: Number of results to return (default 5)
    """
    collection_name = f"workspace_{workspace}"

    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=collection_name,
    )

    results = vectorstore.similarity_search(query, k=num_results)

    # Format results
    context = []
    for r in results:
        file = r.metadata.get("source", "unknown")
        layer = r.metadata.get("layer", "")
        layer_tag = f" [{layer}]" if layer else ""
        context.append(f"### {file}{layer_tag}\n```\n{r.page_content}\n```")

    return "\n\n".join(context)


@mcp.tool()
def search_codebase_smart(
    workspace: str,
    query: str,
    current_file: str = "",
) -> str:
    """Smart search that returns structured results by architecture layer.

    Returns results in 3 categories based on layer roles:
    1. Architecture & Patterns - layers with role "architecture"
    2. Similar Implementations - code in the same layer as current file
    3. Reusable Base Libraries - layers with role "base"

    Args:
        workspace: Name of the workspace collection (e.g., "YourAppName")
        query: What to search for (e.g., "authentication logic")
        current_file: Current file being edited (optional, for context-aware results)
    """
    collection_name = f"workspace_{workspace}"
    layer_map = load_default_layer_map()  # TODO: could load from workspace metadata

    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=collection_name,
    )

    # Detect current layer from file path
    current_layer = None
    if current_file:
        current_layer = classify_layer(current_file, layer_map)

    # 1. Architecture & Patterns - dynamically get layers with "architecture" role
    arch_layers = get_architecture_layers(layer_map)
    architecture = _search_with_layers(vectorstore, query, arch_layers, k=5)

    # 2. Similar Features (same layer if known, otherwise general search)
    if current_layer and current_layer != "other":
        try:
            similar_features = _search_with_filter(vectorstore, query, current_layer, k=10)
        except Exception:
            similar_features = vectorstore.similarity_search(query, k=10)
    else:
        similar_features = vectorstore.similarity_search(query, k=10)

    # 3. Base Libraries - dynamically get layers with "base" role
    base_layer_names = get_base_layers(layer_map)
    base_libs = _search_with_layers(vectorstore, query, base_layer_names, k=8)

    # Format response
    formatted_output = f"""# Codebase Context for: {query}

## Architecture & Patterns
{_format_section(_format_results(architecture))}

## Similar Implementations
{_format_section(_format_results(similar_features))}

## Reusable Base Libraries
{_format_section(_format_results(base_libs))}

---
**Instructions for Claude:**
- Check if base libraries already solve this - REUSE them
- Follow patterns from architecture docs
- Extend similar implementations - don't duplicate
- Only write NEW code for feature-specific logic
"""

    return formatted_output


@mcp.tool()
def list_workspaces() -> str:
    """List all indexed workspace collections."""
    collections = qdrant_client.get_collections()

    workspaces = []
    for collection in collections.collections:
        if collection.name.startswith("workspace_"):
            project_name = collection.name.replace("workspace_", "")
            info = qdrant_client.get_collection(collection.name)
            workspaces.append(f"- {project_name} ({info.points_count} documents)")

    if not workspaces:
        return "No indexed workspaces found."

    return "Indexed workspaces:\n" + "\n".join(workspaces)


@mcp.tool()
def index_workspace(workspace_path: str, force_reindex: bool = False) -> str:
    """Index a workspace directory for semantic code search.

    This indexes all code files in the workspace into a Qdrant collection,
    enabling semantic search via search_codebase and search_codebase_smart.

    Args:
        workspace_path: Absolute path to the workspace directory to index
        force_reindex: If True, delete existing collection and re-index (default False)
    """
    import os

    workspace_path = os.path.abspath(workspace_path)
    project_name = os.path.basename(workspace_path)
    collection_name = f"workspace_{project_name}"

    if not os.path.isdir(workspace_path):
        return f"Error: '{workspace_path}' is not a valid directory."

    # Check if collection already exists
    collections = [c.name for c in qdrant_client.get_collections().collections]
    if collection_name in collections:
        if not force_reindex:
            return (
                f"Collection '{collection_name}' already exists. "
                f"Use force_reindex=True to delete and re-index."
            )
        qdrant_client.delete_collection(collection_name)

    # Load project-specific layer configuration
    layer_map = load_project_layer_map(workspace_path)

    # Load and process documents
    docs = load_workspace(workspace_path, layer_map)
    if not docs:
        return f"No documents found in '{workspace_path}'."

    chunks = split_documents(docs)

    # Create vector store
    create_vectorstore(chunks, collection_name)

    return (
        f"Successfully indexed workspace '{project_name}'.\n"
        f"- Documents: {len(docs)}\n"
        f"- Chunks: {len(chunks)}\n"
        f"- Collection: {collection_name}\n"
        f"- Layers: {list(layer_map.keys())}"
    )


async def main():
    await mcp.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main())