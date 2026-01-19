#!/usr/bin/env python3
import asyncio
from mcp.server.fastmcp import FastMCP
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from config import QDRANT_URL, EMBEDDING_MODEL

mcp = FastMCP("workspace-rag")

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@mcp.tool()
def search_codebase(workspace: str, query: str, num_results: int = 5) -> str:
    """Search indexed workspace for relevant code/files.

    Args:
        workspace: Name of the workspace collection (e.g., "H2BWebApps")
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
        context.append(f"### {file}\n```\n{r.page_content}\n```")

    return "\n\n".join(context)


async def main():
    await mcp.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main())