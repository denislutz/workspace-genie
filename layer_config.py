# layer_config.py - Dynamic layer classification for workspaces
"""
Each project can provide a `.claude/workspace_layers.json` file with custom layer config:

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
        "features": {
          "patterns": ["features/", "modules/"],
          "priority": 80,
          "role": "feature"
        }
      }
    }

Roles:
- "architecture": Documentation, patterns, architecture decisions (highest priority search)
- "base": Reusable utilities, shared code (search for reuse)
- "feature": Feature implementation layers (search for similar implementations)

The priority determines classification order (higher = checked first).
The role determines how layers are grouped in smart search results.

If no config is found, default_layers.json is used.
"""

import fnmatch
import json
from pathlib import Path


def load_default_layer_map() -> dict:
    """Load the default layer map from default_layers.json."""
    default_file = Path(__file__).parent / "default_layers.json"
    try:
        with open(default_file) as f:
            config = json.load(f)
            return config.get("layers", {})
    except Exception as e:
        print(f"Warning: Failed to load default_layers.json: {e}")
        return {}


def load_project_layer_map(workspace_path: str) -> dict:
    """Load project-specific layer map or return default.

    Looks for config at: <workspace>/.claude/workspace_layers.json
    """
    config_file = Path(workspace_path) / ".claude" / "workspace_layers.json"

    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
                if "layers" in config:
                    print(f"Loaded custom layer config from {config_file}")
                    return config["layers"]
        except Exception as e:
            print(f"Warning: Failed to load {config_file}: {e}")

    return load_default_layer_map()


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
