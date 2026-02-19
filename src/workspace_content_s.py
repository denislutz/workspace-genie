#!/usr/bin/env python3
import sys
from pathlib import Path

# Add parent directory to path for config import
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from config import CONTENT_PREVIEW_SIZE


def to_relative_path(abs_path: str, workspace_root: str) -> str:
    """Convert absolute path to relative path."""
    if workspace_root and abs_path.startswith(workspace_root):
        rel = abs_path[len(workspace_root) :]
        return rel.lstrip("/\\")
    return abs_path


def _to_relative_path(abs_path: str, workspace_root: str) -> str:
    """Convert absolute path to relative path."""
    if workspace_root and abs_path.startswith(workspace_root):
        rel = abs_path[len(workspace_root) :]
        return rel.lstrip("/\\")
    return abs_path


def smart_truncate(content: str, max_len: int = CONTENT_PREVIEW_SIZE) -> str:
    """Truncate content at a sensible boundary (newline, semicolon, bracket)."""
    if len(content) <= max_len:
        return content

    # Look for good break points near the limit
    truncated = content[:max_len]

    # Try to break at newline
    last_newline = truncated.rfind("\n")
    if last_newline > max_len * 0.7:  # Don't cut too much
        return truncated[:last_newline]

    # Try to break at semicolon or closing brace
    for char in (";\n", "}\n", ");", "},"):
        pos = truncated.rfind(char)
        if pos > max_len * 0.6:
            return truncated[: pos + len(char)]

    return truncated


def format_results(results, workspace_root: str = "") -> list[str]:
    """Format search results into structured list with relative paths."""
    formatted = []
    seen_content = set()  # Track unique content hashes

    for r in results:
        # Prefer stored relative_path, fallback to computing from source
        rel_path = r.metadata.get("relative_path")
        if not rel_path:
            abs_path = r.metadata.get("source", "unknown")
            stored_root = r.metadata.get("workspace_root", workspace_root)
            rel_path = _to_relative_path(abs_path, stored_root)

        content = smart_truncate(r.page_content)

        # Simple dedup: skip if we've seen very similar content
        content_hash = hash(content[:100])
        if content_hash in seen_content:
            continue
        seen_content.add(content_hash)

        formatted.append(
            {
                "file": rel_path,
                "layer": r.metadata.get("layer", "other"),
                "content": content,
            }
        )

    return formatted


def format_section(items: list, seen_files: set) -> str:
    """Format a section of results as markdown, deduping files across sections."""
    if not items:
        return "_No relevant items found_\n"

    if seen_files is None:
        seen_files = set()

    output = ""
    for item in items:
        file_key = item["file"]
        # Skip if this exact file+content was already shown
        content_key = f"{file_key}:{hash(item['content'][:50])}"
        if content_key in seen_files:
            continue
        seen_files.add(content_key)

        output += f"\n### {item['file']}\n```\n{item['content']}\n```\n"

    return output if output else "_No relevant items found_\n"


__all__ = ["format_results", "format_section", "smart_truncate", "to_relative_path"]
