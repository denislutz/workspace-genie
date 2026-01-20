# index_workspace.py
import os
import sys
from pathlib import Path

import pathspec
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import QDRANT_URL, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from layer_config import load_project_layer_map, classify_layer

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

            # Add layer metadata to each document
            for doc in loaded_docs:
                layer = classify_layer(str(rel_path), layer_map)
                doc.metadata["layer"] = layer
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
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = QdrantVectorStore.from_documents(
        docs,
        embeddings,
        url=QDRANT_URL,
        collection_name=collection_name,
    )
    return vectorstore


def main():
    if len(sys.argv) < 2:
        print("Usage: python index_workspace.py <workspace_path>")
        sys.exit(1)

    workspace_path = sys.argv[1]
    workspace_path = os.path.abspath(workspace_path)
    project_name = os.path.basename(workspace_path)
    collection_name = f"workspace_{project_name}"

    print(f"Indexing workspace: {workspace_path}")
    print(f"Collection name: {collection_name}")

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
