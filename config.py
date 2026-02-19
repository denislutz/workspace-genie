# config.py - Shared configuration for workspace RAG

QDRANT_URL = "http://localhost:6333"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CONTENT_PREVIEW_SIZE = 350  # Max chars to show per chunk in results
