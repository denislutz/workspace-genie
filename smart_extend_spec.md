# Workspace Genie - Setup Guide

## Overview

Ein lokales RAG-System für Codebase-Indexierung mit MCP-Integration für Claude Code in VSCode. Das System ermöglicht kontextbewusstes Code-Verständnis und verhindert Code-Duplikation durch intelligente Wiederverwendung existierender Komponenten.

## Architektur

```
VSCode Workspace
    ↓
[File Watcher] → Erkennt Code-Änderungen
    ↓
[Embedding Pipeline]
├─ Chunking (Functions/Classes)
├─ Embeddings (Ollama - lokal)
└─ Speicherung (Qdrant Collections)
    ↓
[MCP Server] ← Claude Code fragt nach Kontext
├─ Architektur-Wissen
├─ Ähnliche Features
└─ Base Libraries
    ↓
Claude Code nutzt Kontext für intelligentes Coding
```

## Tech Stack

| Component       | Technology                | Warum                                   |
| --------------- | ------------------------- | --------------------------------------- |
| **Embeddings**  | Ollama + nomic-embed-text | Lokal, kostenlos, schnell, GDPR-konform |
| **Vector DB**   | Qdrant                    | Beste Performance/Cost, gute Filterung  |
| **Framework**   | LangChain                 | Industrie-Standard, 10.7% DACH Jobs     |
| **Integration** | MCP Server                | Native Claude Code Integration          |

## Installation

### 1. Basis-Setup

```bash
# Ollama installieren
brew install ollama  # macOS
ollama pull nomic-embed-text

# Qdrant starten
docker run -d -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant

# Python Dependencies
python -m venv venv
source venv/bin/activate
pip install langchain-community langchain-qdrant qdrant-client ollama
```

### 2. Project Structure

```
workspace-genie/
├─ venv/
├─ index_workspace.py          # Indexiert Projekte
├─ query_workspace.py           # Manuelle Queries (Testing)
├─ mcp_workspace_server.py      # MCP Server für Claude Code
├─ list_projects.py             # Zeigt indexierte Projekte
└─ .workspace-genie.json        # Architektur-Config (pro Projekt)
```

## Core Scripts

### index_workspace.py

```python
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import sys
import os

def index_workspace(workspace_path):
    project_name = os.path.basename(workspace_path)

    # Load code files
    loader = DirectoryLoader(
        workspace_path,
        glob="**/*.{ts,tsx,js,jsx,py}",
        show_progress=True,
        exclude=["**/node_modules/**", "**/dist/**", "**/.git/**"]
    )
    docs = loader.load()

    # Add metadata
    for doc in docs:
        doc.metadata["project"] = project_name
        doc.metadata["layer"] = classify_layer(doc.metadata["source"])

    # Create embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    client = QdrantClient(url="http://localhost:6333")

    # Store in project-specific collection
    vectorstore = QdrantVectorStore.from_documents(
        docs,
        embeddings,
        url="http://localhost:6333",
        collection_name=f"workspace_{project_name}"
    )

    print(f"✅ Indexed {len(docs)} files in collection 'workspace_{project_name}'")

def classify_layer(filepath):
    """Klassifiziert Datei nach Architektur-Schicht"""
    if "architecture" in filepath or "docs/" in filepath or "README" in filepath:
        return "architecture"
    if "services/" in filepath:
        return "services"
    if "components/" in filepath or "ui/" in filepath:
        return "components"
    if "utils/" in filepath or "lib/" in filepath or "shared/" in filepath:
        return "utils"
    return "other"

if __name__ == "__main__":
    workspace_path = sys.argv[1]
    index_workspace(workspace_path)
```

**Usage:**

```bash
python index_workspace.py ~/projects/my-react-app
python index_workspace.py ~/projects/backend-api
```

### mcp_workspace_server.py

````python
from mcp.server import Server
from mcp.types import Tool, TextContent
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import OllamaEmbeddings
import json

server = Server("workspace-genie")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="search_codebase",
            description="Search indexed codebase for relevant code in 3 categories: architecture patterns, similar features, and reusable base libraries",
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Project name (e.g., 'my-react-app')"
                    },
                    "query": {
                        "type": "string",
                        "description": "What to search for (e.g., 'authentication logic')"
                    },
                    "current_file": {
                        "type": "string",
                        "description": "Current file being edited (optional, for context)"
                    }
                },
                "required": ["project", "query"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "search_codebase":
        project = arguments['project']
        query = arguments['query']
        current_file = arguments.get('current_file', '')

        # Detect current layer
        current_layer = classify_layer(current_file) if current_file else None

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        collection_name = f"workspace_{project}"

        # 1. Architecture & Patterns (always relevant)
        vectorstore_arch = QdrantVectorStore.from_existing_collection(
            embeddings=embeddings,
            url="http://localhost:6333",
            collection_name=collection_name
        )

        architecture = vectorstore_arch.similarity_search(
            query,
            k=5,
            filter={"layer": "architecture"}
        )

        # 2. Similar Features (same layer if known)
        similar_filter = {"layer": current_layer} if current_layer else {}
        similar_features = vectorstore_arch.similarity_search(
            query,
            k=10,
            filter=similar_filter
        )

        # 3. Base Libraries (utils, shared)
        base_libs = vectorstore_arch.similarity_search(
            query,
            k=8,
            filter={"layer": "utils"}
        )

        # Format response
        response = {
            "architecture": format_results(architecture),
            "similar_features": format_results(similar_features),
            "base_libraries": format_results(base_libs)
        }

        formatted_output = f"""
# Codebase Context for: {query}

## Architecture & Patterns
{format_section(response['architecture'])}

## Similar Implementations
{format_section(response['similar_features'])}

## Reusable Base Libraries
{format_section(response['base_libraries'])}

---
**Instructions for Claude:**
- Check if base libraries already solve this - REUSE them
- Follow patterns from architecture docs
- Extend similar implementations - don't duplicate
- Only write NEW code for feature-specific logic
"""

        return [TextContent(type="text", text=formatted_output)]

def format_results(results):
    return [
        {
            "file": r.metadata["source"],
            "content": r.page_content[:500],  # Erste 500 chars
        }
        for r in results
    ]

def format_section(items):
    if not items:
        return "_No relevant items found_\n"

    output = ""
    for item in items:
        output += f"\n### {item['file']}\n```\n{item['content']}\n```\n"
    return output

def classify_layer(filepath):
    """Same as in index_workspace.py"""
    if "architecture" in filepath or "docs/" in filepath:
        return "architecture"
    if "services/" in filepath:
        return "services"
    if "components/" in filepath:
        return "components"
    if "utils/" in filepath or "shared/" in filepath:
        return "utils"
    return "other"

if __name__ == "__main__":
    import asyncio
    asyncio.run(server.run())
````

### list_projects.py

```python
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
collections = client.get_collections()

print("📚 Indexed Projects:\n")
for collection in collections.collections:
    if collection.name.startswith("workspace_"):
        project_name = collection.name.replace("workspace_", "")
        info = client.get_collection(collection.name)
        print(f"  • {project_name} ({info.points_count} documents)")
```

## Architektur-Config (.workspace-genie.json)

Place in your project root:

```json
{
  "architecture": {
    "layers": [
      {
        "name": "architecture",
        "priority": 10,
        "patterns": ["docs/", "README", "ADR", "architecture/"]
      },
      {
        "name": "services",
        "priority": 8,
        "patterns": ["services/", "src/services/"]
      },
      {
        "name": "components",
        "priority": 7,
        "patterns": ["components/", "ui/", "src/components/"]
      },
      {
        "name": "utils",
        "priority": 9,
        "patterns": ["utils/", "lib/", "shared/", "common/"]
      }
    ],
    "boost_same_layer": 3,
    "boost_base_lib": 2
  }
}
```

## MCP Integration in Claude Code

### 1. MCP Server registrieren

In `~/.config/claude-code/mcp_servers.json`:

```json
{
  "workspace-genie": {
    "command": "python",
    "args": ["/path/to/workspace-genie/mcp_workspace_server.py"],
    "env": {}
  }
}
```

### 2. Server starten

```bash
# In eigenem Terminal
cd workspace-genie
source venv/bin/activate
python mcp_workspace_server.py
```

### 3. In Claude Code nutzen

Claude Code kann jetzt automatisch das Tool `search_codebase` nutzen:

```
You: "Add email validation to UserService"

Claude (intern):
  → Ruft search_codebase(project="my-app", query="email validation", current_file="services/UserService.ts")

  → Bekommt zurück:
     - Architecture: "Use utils for validation"
     - Similar: "AuthService.ts already validates emails"
     - Base: "utils/validation/email.ts exists!"

  → Schreibt Code:
     import { validateEmail } from '../utils/validation/email';
     // Erweitert UserService, schreibt keine neue Validation
```

## Workflow für große Codebases (1M+ LOC)

### Strategie: Hierarchisches Indexing

```python
# index_workspace.py - erweitert

def index_workspace_hierarchical(workspace_path):
    project_name = os.path.basename(workspace_path)

    # Phase 1: Nur wichtigste Files (5-10% der Codebase)
    priority_patterns = [
        "**/*README*",
        "**/*architecture*",
        "**/docs/**/*.md",
        "**/package.json",
        "**/tsconfig.json",
        "**/src/index.ts",
        "**/src/**/index.ts",    # Public APIs
        "**/utils/**/*.ts",       # Base Libs
        "**/shared/**/*.ts",
    ]

    loader = DirectoryLoader(
        workspace_path,
        glob=priority_patterns,
        show_progress=True
    )

    # Rest on-demand oder als separate Collections
```

### Module-basiertes Indexing

```bash
# Indexiere nur spezifische Module
python index_module.py --project=my-app --module=services/UserManagement
python index_module.py --project=my-app --module=components/Dashboard
```

## Performance Tipps

| Codebase Size | Strategy         | Index Time    | RAM   |
| ------------- | ---------------- | ------------- | ----- |
| < 50k LOC     | Alles indexieren | 5-10min       | 1-2GB |
| 50k-500k LOC  | Hierarchisch     | 10-30min      | 2-4GB |
| > 500k LOC    | Module on-demand | 5-15min/Modul | 1-2GB |

## Testing

### Manueller Test (ohne MCP)

```bash
# 1. Indexiere Projekt
python index_workspace.py ~/projects/my-app

# 2. Test Query
python query_workspace.py my-app "authentication logic"

# 3. Prüfe Ergebnisse
python list_projects.py
```

### MCP Test

```bash
# Terminal 1: Start MCP Server
python mcp_workspace_server.py

# Terminal 2: Test mit Claude Code
# Öffne VSCode, aktiviere Claude Code, frage:
"Search my codebase for email validation logic"
```

## Troubleshooting

### Qdrant Connection Error

```bash
# Prüfe ob Qdrant läuft
docker ps | grep qdrant

# Neu starten
docker restart <qdrant-container-id>
```

### Ollama Model nicht gefunden

```bash
ollama list
ollama pull nomic-embed-text
```

### Langsame Queries

```python
# Reduziere k (Anzahl Ergebnisse)
results = vectorstore.similarity_search(query, k=3)  # statt k=10

# Oder: Nutze score_threshold
results = vectorstore.similarity_search(
    query,
    k=5,
    score_threshold=0.7  # nur relevante
)
```

## Nächste Schritte

1. **Indexiere dein erstes Projekt** mit `index_workspace.py`
2. **Teste manuelle Queries** mit `query_workspace.py`
3. **Starte MCP Server** und integriere in Claude Code
4. **Erstelle `.workspace-genie.json`** für deine Architektur
5. **Erweitere für große Codebases** mit hierarchischem Indexing

## Vorteile dieses Setups

✅ **Lokal & GDPR-konform** - Kein Code verlässt deinen Rechner  
✅ **Schnell** - Ollama embeddings in Millisekunden  
✅ **Kostenlos** - Keine API-Kosten  
✅ **Job-relevant** - LangChain + Qdrant = DACH Markt-Standard  
✅ **Smart** - LLM entscheidet über Code-Reuse, nicht Algorithmus  
✅ **Skalierbar** - Funktioniert von 10k bis 1M+ LOC

## Resources

- Qdrant Docs: <https://qdrant.tech/documentation/>
- LangChain Qdrant: <https://python.langchain.com/docs/integrations/vectorstores/qdrant>
- Ollama Models: <https://ollama.ai/library>
- MCP Protocol: <https://modelcontextprotocol.io/>

- Qdrant Docs: <https://qdrant.tech/documentation/>
- LangChain Qdrant: <https://python.langchain.com/docs/integrations/vectorstores/qdrant>
- Ollama Models: <https://ollama.ai/library>
- MCP Protocol: <https://modelcontextprotocol.io/>
