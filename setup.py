#!/usr/bin/env python3
"""Setup workspace-genie for a target repository."""
import json
import os
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python setup.py <target_repo_path>")
        sys.exit(1)

    target_repo = os.path.abspath(sys.argv[1])
    workspace_genie_dir = os.path.dirname(os.path.abspath(__file__))
    project_name = os.path.basename(target_repo)

    # Check for layer config
    layer_config_path = Path(target_repo) / ".claude" / "workspace_layers.json"
    if not layer_config_path.exists():
        print(f"No layer config found at: {layer_config_path}")
        print()
        print("You can create a custom layer config to improve search results.")
        print("Example .claude/workspace_layers.json:")
        print()
        print("""{
  "layers": {
    "docs": {
      "patterns": ["docs/", "README", "architecture/"],
      "priority": 100,
      "role": "architecture"
    },
    "core": {
      "patterns": ["core/", "lib/"],
      "priority": 90,
      "role": "base"
    },
    "features": {
      "patterns": ["features/", "modules/"],
      "priority": 80,
      "role": "feature"
    }
  }
}""")
        print()
        response = input("Proceed with default layer map? [Y/n]: ").strip().lower()
        if response == "n":
            print(f"\nCreate your config at: {layer_config_path}")
            print("Then run this command again.")
            sys.exit(0)
        print("\nUsing default layer map...")
    else:
        print(f"Found layer config: {layer_config_path}")

    # Step 1: Index the codebase (check if collection exists first)
    import subprocess
    from qdrant_client import QdrantClient

    collection_name = f"workspace_{project_name}"
    client = QdrantClient(url="http://localhost:6333")
    collections = [c.name for c in client.get_collections().collections]

    should_index = True
    if collection_name in collections:
        response = input(f"Collection '{collection_name}' already exists. Delete and re-index? [Y/n]: ").strip().lower()
        if response == "n":
            print("Skipping indexing, keeping existing collection...")
            should_index = False
        else:
            client.delete_collection(collection_name)
            print(f"Deleted existing collection: {collection_name}")

    if should_index:
        print(f"\nIndexing {target_repo}...")
        # Use --skip-delete flag to skip the deletion prompt in index_workspace.py
        result = subprocess.run(
            ["python", f"{workspace_genie_dir}/index_workspace.py", target_repo, "--skip-delete-check"],
            cwd=workspace_genie_dir
        )
        if result.returncode != 0:
            print("Indexing failed!")
            sys.exit(1)

    # Step 2: Configure MCP in target repo
    mcp_config_path = Path(target_repo) / ".mcp.json"

    # Load existing config or create new
    if mcp_config_path.exists():
        with open(mcp_config_path) as f:
            config = json.load(f)
    else:
        config = {"mcpServers": {}}

    # Add/update workspace-genie server
    config["mcpServers"]["workspace-genie"] = {
        "type": "stdio",
        "command": "mise",
        "args": [
            "x",
            "-C",
            workspace_genie_dir,
            "--",
            "python",
            f"{workspace_genie_dir}/mcp_workspace.py"
        ],
        "env": {}
    }

    with open(mcp_config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSetup complete!")
    if should_index:
        print(f"  - Indexed: workspace_{project_name}")
    else:
        print(f"  - Using existing index: workspace_{project_name}")
    print(f"  - MCP config: {mcp_config_path}")
    print(f"\nRestart Claude Code in {target_repo} to use workspace-genie.")


if __name__ == "__main__":
    main()
