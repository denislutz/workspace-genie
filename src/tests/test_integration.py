#!/usr/bin/env python3
"""
Real MCP integration tests using dependency injection.
Tests the actual MCP tools end-to-end with in-memory Qdrant.
"""

import tempfile
import unittest
from pathlib import Path
import sys
import shutil
import os

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client.http.models import Distance, VectorParams

# Import the configuration system
from vector_store_config import (
    VectorStoreConfig,
    set_vector_store_config,
    reset_vector_store_config,
)

# Import MCP modules AFTER importing config system
import mcp_workspace
import vector_store_s


class TestMCPDependencyInjection(unittest.TestCase):
    """Real integration tests for MCP workspace tools using dependency injection."""

    @classmethod
    def setUpClass(cls):
        """Set up shared test resources."""
        # Create in-memory Qdrant client and embeddings
        cls.qdrant_client = QdrantClient(":memory:")
        cls.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Create test workspace
        cls.test_workspace_dir = tempfile.mkdtemp()
        cls.workspace_name = os.path.basename(cls.test_workspace_dir)
        cls._create_test_workspace()

        # Inject test configuration
        cls._inject_test_config()

    @classmethod
    def tearDownClass(cls):
        """Clean up shared resources."""
        reset_vector_store_config()  # Reset global config
        shutil.rmtree(cls.test_workspace_dir, ignore_errors=True)

    def setUp(self):
        """Set up for each test."""
        # Clear collections
        try:
            collections = self.qdrant_client.get_collections()
            for collection in collections.collections:
                self.qdrant_client.delete_collection(collection.name)
        except Exception:
            pass

        # Clear caches
        vector_store_s._workspace_roots.clear()

    @classmethod
    def _inject_test_config(cls):
        """Inject test configuration with in-memory clients."""
        test_config = VectorStoreConfig(
            qdrant_client=cls.qdrant_client,
            embeddings=cls.embeddings,
        )
        set_vector_store_config(test_config)

        # Reload modules to pick up new configuration
        import importlib

        importlib.reload(vector_store_s)
        importlib.reload(mcp_workspace)

    @classmethod
    def _create_test_workspace(cls):
        """Create a realistic test workspace."""
        workspace_path = Path(cls.test_workspace_dir)

        # Create React component
        (workspace_path / "src" / "components").mkdir(parents=True, exist_ok=True)
        (workspace_path / "src" / "components" / "Button.jsx").write_text(
            """
import React from 'react';

const Button = ({ children, onClick }) => {
    return (
        <button onClick={onClick}>
            {children}
        </button>
    );
};

export default Button;
"""
        )

        # Create Python utility
        (workspace_path / "src" / "utils").mkdir(parents=True, exist_ok=True)
        (workspace_path / "src" / "utils" / "auth.py").write_text(
            """
def hash_password(password):
    return f"hashed_{password}"

def verify_password(password, hashed):
    return hash_password(password) == hashed

class UserAuth:
    def authenticate(self, username, password):
        if username == "admin":
            return {"id": 1, "name": "Admin"}
        return None
"""
        )

        # Create test file
        (workspace_path / "tests").mkdir(parents=True, exist_ok=True)
        (workspace_path / "tests" / "test_auth.py").write_text(
            """
import unittest
from src.utils.auth import hash_password, verify_password

class TestAuth(unittest.TestCase):
    def test_hash_password(self):
        result = hash_password("test123")
        self.assertIn("hashed_test123", result)
        
    def test_verify_password(self):
        self.assertTrue(verify_password("test123", "hashed_test123"))
"""
        )

        # Create package.json
        (workspace_path / "package.json").write_text(
            """
{
  "name": "test-workspace",
  "version": "1.0.0",
  "dependencies": {
    "react": "^18.0.0"
  }
}
"""
        )

    def test_index_workspace_mcp_tool(self):
        """Test the index_workspace MCP tool with dependency injection."""
        result = mcp_workspace.index_workspace(
            self.test_workspace_dir, force_reindex=False
        )

        # Verify success
        self.assertIn("Successfully indexed workspace", result)
        self.assertIn("Documents:", result)
        self.assertIn("Chunks:", result)

        # Verify collection exists in our in-memory client
        collections = self.qdrant_client.get_collections()
        workspace_collections = [
            c.name for c in collections.collections if c.name.startswith("workspace_")
        ]
        self.assertGreater(len(workspace_collections), 0)

    def test_search_codebase_mcp_tool(self):
        """Test the search_codebase MCP tool with dependency injection."""
        # Index workspace first
        mcp_workspace.index_workspace(self.test_workspace_dir, force_reindex=False)

        # Test search
        result = mcp_workspace.search_codebase(self.workspace_name, "React button", 3)

        # Should find Button component
        self.assertIn("Button.jsx", result)
        self.assertIn("React", result)
        self.assertIn("onClick", result)

    def test_search_codebase_smart_mcp_tool(self):
        """Test the search_codebase_smart MCP tool with dependency injection."""
        # Index workspace first
        mcp_workspace.index_workspace(self.test_workspace_dir, force_reindex=False)

        # Test smart search
        result = mcp_workspace.search_codebase_smart(
            self.workspace_name, "user authentication", "src/utils/auth.py"
        )

        # Should return structured results
        self.assertIn("## Architecture & Patterns", result)
        self.assertIn("## Similar Implementations", result)
        self.assertIn("## Reusable Base Libraries", result)
        self.assertIn("Context: user authentication", result)

    def test_list_workspaces_mcp_tool(self):
        """Test the list_workspaces MCP tool with dependency injection."""
        # Index workspace first
        mcp_workspace.index_workspace(self.test_workspace_dir, force_reindex=False)

        # List workspaces
        result = mcp_workspace.list_workspaces()

        # Should show our workspace
        self.assertIn("Indexed workspaces:", result)
        self.assertIn(self.workspace_name, result)

    def test_find_similar_files_mcp_tool(self):
        """Test the find_similar_files MCP tool with dependency injection."""
        # Index workspace first
        mcp_workspace.index_workspace(self.test_workspace_dir, force_reindex=False)

        # Test finding similar files
        result = mcp_workspace.find_similar_files(
            task="implement user login",
            current_file="src/components/Login.jsx",
            workspace=self.workspace_name,
        )

        # Should return structured guidance
        self.assertIn("task", result)
        self.assertIn("similar_files", result)
        self.assertIn("recommendation", result)
        self.assertIn("next_step", result)
        self.assertEqual(result["task"], "implement user login")

    def test_find_similar_patterns_mcp_tool(self):
        """Test the find_similar_patterns MCP tool with dependency injection."""
        # Index workspace first
        mcp_workspace.index_workspace(self.test_workspace_dir, force_reindex=False)

        # Test finding similar patterns
        result = mcp_workspace.find_similar_patterns(
            workspace=self.workspace_name,
            query="function implementation",
            current_file="src/utils/helper.py",
        )

        # Should return layer-specific patterns
        self.assertIn("current_layer", result)
        self.assertIn("similar_patterns", result)
        self.assertIn("usage_hint", result)

    def test_workspace_root_caching(self):
        """Test workspace root path caching with dependency injection."""
        # Index workspace first
        mcp_workspace.index_workspace(self.test_workspace_dir, force_reindex=False)

        # First call should compute and cache
        root1 = vector_store_s.get_workspace_root(self.workspace_name)

        # Second call should use cache
        root2 = vector_store_s.get_workspace_root(self.workspace_name)

        self.assertEqual(root1, root2)
        self.assertEqual(root1, self.test_workspace_dir)

    def test_error_handling(self):
        """Test error handling in MCP tools."""
        # Test invalid workspace path
        result = mcp_workspace.index_workspace("/nonexistent/path", force_reindex=False)
        self.assertIn("Error:", result)

        # Test searching non-existent workspace (should handle gracefully)
        try:
            result = mcp_workspace.search_codebase("nonexistent", "test query", 3)
            # Should not crash - may return empty or error
        except Exception as e:
            # If it throws, should be a reasonable error
            error_str = str(e)
            self.assertTrue(
                "404" in error_str
                or "not found" in error_str.lower()
                or "Collection" in error_str
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
