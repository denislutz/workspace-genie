# Workspace Genie

A RAG (Retrieval-Augmented Generation) system for indexing and querying workspace codebases. Provides semantic code search via MCP (Model Context Protocol) to external LLMs like Claude.

## Stack

- Python 3.11 (managed via mise)
- Qdrant vector database for storage
- LangChain for document processing
- sentence-transformers for local embeddings (all-MiniLM-L6-v2)
- MCP (Model Context Protocol) for LLM integration

## Setup

```bash
mise install          # Install Python 3.11
mise run install      # Install dependencies
mise run qdrant       # Start Qdrant (separate terminal)
```

## Usage

### Index a workspace

```bash
mise run index <workspace_path>
```

This indexes the workspace into a Qdrant collection named `workspace_<project_name>`.

### Query via CLI

```bash
mise run query workspace_<name> "search query"
```

### MCP Server

The MCP server exposes workspace search to external LLMs (e.g., Claude Desktop, Claude Code).

```bash
python mcp_workspace.py
```

Add to Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "workspace-genie": {
      "command": "python",
      "args": ["/path/to/mcp_workspace.py"]
    }
  }
}
```

The server provides these tools:

### `search_codebase`

Basic search returning flat results:

- `workspace`: Collection name (e.g., "YourAppName")
- `query`: Semantic search query
- `num_results`: Number of results (default 5)

### `search_codebase_smart`

Smart search returning structured results by architecture layer:

- `workspace`: Collection name
- `query`: What to search for
- `current_file`: Current file being edited (optional, for context-aware results)

Returns results in 3 categories:

1. **Architecture & Patterns** - docs, READMEs, architecture decisions
2. **Similar Implementations** - code in the same layer as current file
3. **Reusable Base Libraries** - utils, shared code, helpers

### `list_workspaces`

Lists all indexed workspace collections.

## Architecture

- `index_workspace.py` - Indexes workspace files into Qdrant vectors
  - Respects `.gitignore` patterns
  - Excludes node_modules, binaries, lock files
  - Chunks documents for better retrieval
  - Adds layer metadata (architecture, services, components, utils)
- `query_workspace.py` - CLI tool to search indexed workspaces
  - Supports `--layer` filter for layer-specific queries
- `mcp_workspace.py` - MCP server exposing search to LLMs
- `layer_config.py` - Dynamic layer classification system
- `config.py` - Shared configuration

## Layer Classification

Files are classified into layers based on path patterns:

- `architecture` - docs/, README, architecture/
- `services` - services/, api/
- `components` - components/, ui/, views/
- `utils` - utils/, lib/, shared/, common/, helpers/

### Custom Layer Config

Projects can override classification by creating `.claude/workspace_layers.json`:

```json
{
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
    "modules": {
      "patterns": ["modules/", "features/"],
      "priority": 80,
      "role": "feature"
    },
    "helpers": {
      "patterns": ["helpers/", "utils/"],
      "priority": 85,
      "role": "base"
    }
  }
}
```

**Layer Config Fields:**

- `patterns`: Path patterns to match files (substring or glob)
- `priority`: Classification order (higher = checked first)
- `role`: Groups layers in smart search results:
  - `"architecture"` → Architecture & Patterns section
  - `"base"` → Reusable Base Libraries section
  - `"feature"` → Similar Implementations section

## Mise Tasks

| Task                                  | Description                                |
| ------------------------------------- | ------------------------------------------ |
| `mise run install`                    | Install dependencies                       |
| `mise run qdrant`                     | Start Qdrant database                      |
| `mise run setupForProject <path>`     | Index a repo + configure MCP (recommended) |
| `mise run index <path>`               | Index a workspace                          |
| `mise run query <collection> <query>` | Search indexed content                     |
| `mise run list-collections`           | List all collections                       |
| `mise run collection-info <name>`     | Get collection stats                       |
| `mise run delete-collection <name>`   | Delete a collection                        |

## Configuration

The system uses environment variables for configuration. Copy `.env.example` to `.env` and adjust as needed:

```bash
cp .env.example .env
```

### Environment Variables

#### Workspace Configuration

- `WORKSPACE` - Default workspace name (also used as the index name)

#### LLM Configuration

- `LLM_MODEL` - Which LLM model to use
  - Options: `ollama`, `ollama-coder`, `ollama-large`, `claude`
  - Default: `ollama`

#### Service Configuration

- `OLLAMA_BASE_URL` - Ollama server URL
  - Default: `http://localhost:11434`
- `QDRANT_URL` - Qdrant server URL  
  - Default: `http://localhost:6333`

#### Debug Configuration

- `DEBUG_AGENT` - Enable detailed debug logging
  - Options: `true`, `false`
  - Default: `false`

### Usage Examples

```bash
# Use a specific workspace
export WORKSPACE=my-project && mise run agent

# Enable debug logging
export DEBUG_AGENT=true && mise run agent

# Use different LLM
export LLM_MODEL=claude && mise run agent

# Start required services
ollama serve &                    # Start Ollama
mise run qdrant-start              # Start Qdrant
mise run index /path/to/workspace   # Index workspace
```

## Requirements

- Docker (for running Qdrant)
- mise (for Python version and task management)
