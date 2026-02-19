"""
Vector store configuration with dependency injection.
Supports environment variables and test injection.
"""

import os
from typing import Optional
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore


class VectorStoreConfig:
    """Configuration for vector store with dependency injection support."""
    
    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
        qdrant_client: Optional[QdrantClient] = None,
        embeddings: Optional[HuggingFaceEmbeddings] = None,
    ):
        # Allow injection of dependencies (for testing)
        self._qdrant_client = qdrant_client
        self._embeddings = embeddings
        
        # Use provided values or fall back to environment/config
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    @property
    def qdrant_client(self) -> QdrantClient:
        """Get Qdrant client (create if not injected)."""
        if self._qdrant_client is not None:
            return self._qdrant_client
        
        # Create client based on URL
        if self.qdrant_url == ":memory:":
            return QdrantClient(":memory:")
        else:
            return QdrantClient(url=self.qdrant_url)
    
    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Get embeddings model (create if not injected)."""
        if self._embeddings is not None:
            return self._embeddings
        
        return HuggingFaceEmbeddings(model_name=self.embedding_model)
    
    def create_vectorstore(
        self, 
        collection_name: str,
        client: Optional[QdrantClient] = None,
        embeddings: Optional[HuggingFaceEmbeddings] = None
    ) -> QdrantVectorStore:
        """Create a vector store with the given configuration."""
        # Use injected clients or fall back to defaults
        qdrant_client = client or self.qdrant_client
        embeddings_model = embeddings or self.embeddings
        
        return QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection_name,
            embedding=embeddings_model,
        )


# Global configuration instance (can be overridden for testing)
_global_config: Optional[VectorStoreConfig] = None


def get_vector_store_config() -> VectorStoreConfig:
    """Get the global vector store configuration."""
    global _global_config
    if _global_config is None:
        _global_config = VectorStoreConfig()
    return _global_config


def set_vector_store_config(config: VectorStoreConfig) -> None:
    """Set the global vector store configuration (for testing)."""
    global _global_config
    _global_config = config


def reset_vector_store_config() -> None:
    """Reset the global vector store configuration."""
    global _global_config
    _global_config = None


# Convenience functions that use the global config
def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client using global configuration."""
    return get_vector_store_config().qdrant_client


def get_embeddings() -> HuggingFaceEmbeddings:
    """Get embeddings using global configuration."""
    return get_vector_store_config().embeddings


def create_vectorstore(collection_name: str) -> QdrantVectorStore:
    """Create vector store using global configuration."""
    return get_vector_store_config().create_vectorstore(collection_name)
