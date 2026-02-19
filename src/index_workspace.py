# index_workspace.py
import os
import sys
from pathlib import Path

import pathspec
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

import sys
import os
from pathlib import Path

# Add parent directory to path for config import
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from config import CHUNK_SIZE, CHUNK_OVERLAP
from layer_config import load_project_layer_map, classify_layer
from vector_store_config import get_qdrant_client, get_embeddings

# Always exclude these patterns (in addition to .gitignore)
DEFAULT_EXCLUDES = [
    "node_modules/",
    ".git/",
    "__pycache__/",
    ".venv/",
    "venv/",
    "dist/",
    "build/",
    ".next/",
    ".nuxt/",
    "*.ico",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.svg",
    "*.woff",
    "*.woff2",
    "*.ttf",
    "*.eot",
    "*.pdf",
    "*.zip",
    "*.tar.gz",
    "*.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
]


def load_gitignore_spec(workspace_path: str) -> pathspec.PathSpec:
    """Load .gitignore patterns and combine with default excludes."""
    patterns = DEFAULT_EXCLUDES.copy()

    gitignore_path = Path(workspace_path) / ".gitignore"
    if gitignore_path.exists():
        with open(gitignore_path, "r") as f:
            gitignore_patterns = f.read().splitlines()
            patterns.extend(gitignore_patterns)

    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def load_workspace(workspace_path: str, layer_map: dict):
    """Load documents from workspace directory, respecting .gitignore and adding layer metadata."""
    workspace = Path(workspace_path)
    workspace_root = str(workspace.resolve())
    spec = load_gitignore_spec(workspace_path)

    docs = []
    files = list(workspace.rglob("*"))
    total = len(files)

    layer_counts = {}

    for i, file_path in enumerate(files):
        if not file_path.is_file():
            continue

        # Get relative path for matching
        rel_path = file_path.relative_to(workspace)

        # Skip if matches gitignore patterns
        if spec.match_file(str(rel_path)):
            continue

        try:
            loader = TextLoader(str(file_path), autodetect_encoding=True)
            loaded_docs = loader.load()

            # Add metadata to each document
            for doc in loaded_docs:
                layer = classify_layer(str(rel_path), layer_map)
                doc.metadata["layer"] = layer
                doc.metadata["workspace_root"] = workspace_root
                doc.metadata["relative_path"] = str(rel_path)
                layer_counts[layer] = layer_counts.get(layer, 0) + 1

            docs.extend(loaded_docs)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

        if (i + 1) % 100 == 0:
            print(f"Scanned {i + 1}/{total} files...")

    print(f"Layer distribution: {layer_counts}")
    return docs


def split_documents(docs):
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def create_vectorstore(docs, collection_name: str):
    """Create Qdrant vector store from documents."""
    embeddings = get_embeddings()
    qdrant_client = get_qdrant_client()

    # Create collection manually if it doesn't exist
    from qdrant_client.http.models import Distance, VectorParams

    try:
        qdrant_client.get_collection(collection_name)
    except Exception:
        # Collection doesn't exist, create it
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    # Create vector store with existing client
    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    # Add documents
    vectorstore.add_documents(docs)
    return vectorstore


def main():
    if len(sys.argv) < 2:
        print("Usage: python index_workspace.py <workspace_path> [--skip-delete-check]")
        sys.exit(1)

    workspace_path = sys.argv[1]
    workspace_path = os.path.abspath(workspace_path)
    skip_delete_check = "--skip-delete-check" in sys.argv
    project_name = os.path.basename(workspace_path)
    collection_name = f"workspace_{project_name}"

    print(f"Indexing workspace: {workspace_path}")
    print(f"Collection name: {collection_name}")

    # Check if collection exists and ask before deletion (unless skipped)
    if not skip_delete_check:
        client = get_qdrant_client()
        collections = [c.name for c in client.get_collections().collections]
        if collection_name in collections:
            response = (
                input(
                    f"Collection '{collection_name}' already exists. Delete and re-index? [Y/n]: "
                )
                .strip()
                .lower()
            )
            if response == "n":
                print("Aborted.")
                sys.exit(0)
            client.delete_collection(collection_name)
            print(f"Deleted existing collection: {collection_name}")

    # Load project-specific layer configuration
    layer_map = load_project_layer_map(workspace_path)
    print(f"Using {len(layer_map)} layers: {list(layer_map.keys())}")

    # Load and process documents with layer metadata
    docs = load_workspace(workspace_path, layer_map)
    print(f"Loaded {len(docs)} documents")

    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    # Create vector store
    vectorstore = create_vectorstore(chunks, collection_name)
    print(f"Created vector store with collection: {collection_name}")

    return vectorstore


if __name__ == "__main__":
    main()
