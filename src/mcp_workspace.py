#!/usr/bin/env python3
import asyncio
import logging
from typing import Any
from mcp.server.fastmcp import FastMCP


import sys
import os
from pathlib import Path

from workspace_content_s import format_results, format_section

# Add parent directory to path for config import
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from config import CONTENT_PREVIEW_SIZE
from layer_config import (
    classify_layer,
    load_default_layer_map,
    get_architecture_layers,
    get_base_layers,
    load_project_layer_map,
)
from index_workspace import load_workspace, split_documents, create_vectorstore
from vector_store_s import *
from vector_store_config import get_qdrant_client, get_embeddings
import vector_store_s

# Module-level setup
logger = logging.getLogger(__name__)
mcp = FastMCP("workspace-genie")

# Cache for workspace root paths (collection_name -> root_path)
_workspace_roots: dict[str, str] = {}

# Use dependency injection for clients
embeddings = get_embeddings()
qdrant_client = get_qdrant_client()


def _search_with_layers(workspace: str, query: str, layers: list[str], k: int):
    """Search across multiple layers, distributing k across them."""
    if not layers:
        return []

    results = []
    k_per_layer = max(1, k // len(layers))

    for layer in layers:
        try:
            layer_results = search_with_filter(
                workspace=workspace,
                query=query,
                filter=layer,
                k=k_per_layer,
            )
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

    vectorstore = vector_store_s.store_for_workspace(workspace)

    results = vectorstore.similarity_search(query, k=num_results)

    workspace_root = vector_store_s.get_workspace_root(workspace)
    formatted = format_results(results, workspace_root)

    # Format results with relative paths
    context = []
    for item in formatted:
        layer_tag = f" [{item['layer']}]" if item["layer"] else ""
        context.append(f"### {item['file']}{layer_tag}\n```\n{item['content']}\n```")

    return "\n\n".join(context)


def to_workspace_id(workspace_id: str) -> str:
    """Creates a workspace id out of a given unique workspace name for unique index"""
    if "/" in workspace_id:
        workspace_id_name = workspace_id.split("/")[-1]
    else:
        workspace_id_name = workspace_id

    return f"workspace_{workspace_id_name}"


@mcp.tool()
def find_similar_files(
    *,
    task: str = "implement user login",
    current_file: str = "",
    workspace: str = "workspace-genie",
) -> dict[str, str | list[str]]:
    """Agentic search: finds context + examples + suggests approach for a task."""
    similar_raw = search_codebase_smart(workspace, task, current_file)
    similar_files: list[str] = []
    for line in similar_raw.split("\n"):
        if line.startswith("### apps/"):
            similar_files.append(line[4:])

    return {
        "task": task,
        "current_file": current_file,
        "similar_files": similar_files[:5],
        "recommendation": f"Use patterns from {similar_files[:5] if similar_files else 'base libs'}",
        "next_step": "Open the top similar file and adapt the pattern",
    }


@mcp.tool()
def find_similar_patterns(
    workspace: str, query: str, current_file: str = ""
) -> dict[str, Any]:
    """Find similar patterns in the same layer as current file."""

    # Reuse your existing layer detection
    layer = classify_layer(current_file, load_default_layer_map())

    # Search within that layer only
    results = search_with_filter(workspace=workspace, query=query, filter=layer, k=8)

    similar_patters = [
        {"file": r.metadata.get("source", "unknown"), "snippet": r.page_content[:200]}
        for r in results
    ]

    return {
        "current_layer": layer,
        "similar_patterns": similar_patters,
        "usage_hint": f"Follow the {layer} layer conventions shown above",
    }


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

    layer_map = load_default_layer_map()  # TODO: could load from workspace metadata

    vectorstore = vector_store_s.store_for_workspace(workspace)

    # Detect current layer from file path
    current_layer = None
    if current_file:
        current_layer = classify_layer(current_file, layer_map)

    # 1. Architecture & Patterns - dynamically get layers with "architecture" role
    arch_layers = get_architecture_layers(layer_map)
    architecture = _search_with_layers(
        workspace=workspace, query=query, layers=arch_layers, k=5
    )

    # 2. Similar Features (same layer if known, otherwise general search)
    if current_layer and current_layer != "other":
        try:
            similar_features = search_with_filter(
                workspace=workspace,
                query=query,
                filter=current_layer,
                k=10,
            )
        except Exception:
            similar_features = vectorstore.similarity_search(query, k=10)
    else:
        similar_features = vectorstore.similarity_search(query, k=10)

    # 3. Base Libraries - dynamically get layers with "base" role
    base_layer_names = get_base_layers(layer_map)
    base_libs = _search_with_layers(workspace, query, base_layer_names, k=8)

    # Format with deduplication across sections
    seen_files: set = set()
    workspace_root = vector_store_s.get_workspace_root(workspace)
    arch_formatted = format_results(architecture, workspace_root)
    similar_formatted = format_results(similar_features, workspace_root)
    base_formatted = format_results(base_libs, workspace_root)

    # Format response with cross-section deduplication
    formatted_output = f"""# Context: {query}

    ## Architecture & Patterns
    {format_section(arch_formatted, seen_files)}
    ## Similar Implementations
    {format_section(similar_formatted, seen_files)}
    ## Reusable Base Libraries
    {format_section(base_formatted, seen_files)}
    ---
    **Guidance:** Reuse base libs, follow arch patterns, extend similar code."""

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
    collection_name = to_workspace_id(project_name)

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
