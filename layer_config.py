# layer_config.py - Dynamic layer classification for workspaces
"""
Each project provides a `.workspace_layers.py` file in its root with:

    LAYER_MAP = {
        "architecture": {
            "patterns": ["docs/", "README", "architecture/"],
            "priority": 10,
            "role": "architecture",  # highest priority - docs, patterns
        },
        "services": {
            "patterns": ["services/", "api/"],
            "priority": 8,
            "role": "feature",  # feature implementation layers
        },
        "components": {
            "patterns": ["components/", "ui/"],
            "priority": 7,
            "role": "feature",
        },
        "utils": {
            "patterns": ["utils/", "lib/", "shared/"],
            "priority": 9,
            "role": "base",  # reusable base libraries
        },
    }

Roles:
- "architecture": Documentation, patterns, architecture decisions (highest priority search)
- "base": Reusable utilities, shared code (search for reuse)
- "feature": Feature implementation layers (search for similar implementations)

The priority determines classification order (higher = checked first).
The role determines how layers are grouped in smart search results.

If no config is found, default classification is used.
"""

import fnmatch
import importlib.util
from pathlib import Path
from typing import Optional

# Default layer map used when project has no custom config
DEFAULT_LAYER_MAP = {
    "architecture": {
        "patterns": ["**/docs/**", "**/README*", "**/architecture/**", "**/*.md", "**/ADR/**"],
        "priority": 10,
        "role": "architecture",
    },
    "services": {
        "patterns": ["**/services/**", "**/api/**", "**/src/services/**"],
        "priority": 8,
        "role": "feature",
    },
    "components": {
        "patterns": ["**/components/**", "**/ui/**", "**/src/components/**", "**/views/**"],
        "priority": 7,
        "role": "feature",
    },
    "utils": {
        "patterns": ["**/utils/**", "**/lib/**", "**/shared/**", "**/common/**", "**/helpers/**"],
        "priority": 9,
        "role": "base",
    },
}


def load_project_layer_map(workspace_path: str) -> dict:
    """Load project-specific layer map or return default."""
    config_file = Path(workspace_path) / ".workspace_layers.py"

    if config_file.exists():
        try:
            spec = importlib.util.spec_from_file_location("workspace_layers", config_file)
            if spec is None or spec.loader is None:
                print(f"Warning: Could not load spec for {config_file}")
                return DEFAULT_LAYER_MAP

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, "LAYER_MAP"):
                print(f"Loaded custom layer config from {config_file}")
                return module.LAYER_MAP
        except Exception as e:
            print(f"Warning: Failed to load {config_file}: {e}")

    return DEFAULT_LAYER_MAP


def classify_layer(filepath: str, layer_map: dict) -> str:
    """Classify a file path into an architecture layer based on the layer map."""
    filepath_normalized = filepath.replace("\\", "/")

    # Check each layer's patterns (sorted by priority, highest first)
    sorted_layers = sorted(
        layer_map.items(),
        key=lambda x: x[1].get("priority", 0),
        reverse=True
    )

    for layer_name, config in sorted_layers:
        patterns = config.get("patterns", [])
        for pattern in patterns:
            if fnmatch.fnmatch(filepath_normalized, pattern):
                return layer_name
            # Also check if pattern appears as substring for simple patterns like "services/"
            if not any(c in pattern for c in "*?[]") and pattern.rstrip("/") in filepath_normalized:
                return layer_name

    return "other"


def get_layer_priority(layer: str, layer_map: dict) -> int:
    """Get priority for a layer."""
    return layer_map.get(layer, {}).get("priority", 0)


def get_layers_by_role(layer_map: dict, role: str) -> list[str]:
    """Get all layer names that have the specified role."""
    return [
        name for name, config in layer_map.items()
        if config.get("role") == role
    ]


def get_architecture_layers(layer_map: dict) -> list[str]:
    """Get layers with 'architecture' role (docs, patterns, decisions)."""
    return get_layers_by_role(layer_map, "architecture")


def get_base_layers(layer_map: dict) -> list[str]:
    """Get layers with 'base' role (reusable utilities, shared code)."""
    return get_layers_by_role(layer_map, "base")


def get_feature_layers(layer_map: dict) -> list[str]:
    """Get layers with 'feature' role (implementation layers)."""
    return get_layers_by_role(layer_map, "feature")
