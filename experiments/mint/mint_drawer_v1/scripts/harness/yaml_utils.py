"""
Harness v1.3 — YAML utilities for sovereign file I/O.
Atomic writes: write to temp file, then rename.
"""
import os
import sys
import yaml
import shutil
from pathlib import Path
from typing import Any, Dict

CAMPAIGN_ROOT = Path(__file__).parent.parent.parent.resolve()
SOVEREIGN_DIR = CAMPAIGN_ROOT / "sovereign"
sys.path.insert(0, str(CAMPAIGN_ROOT / "scripts" / "harness"))


def load_yaml(filename: str) -> Dict[str, Any]:
    path = SOVEREIGN_DIR / filename
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(filename: str, data: Dict[str, Any]) -> None:
    """Atomic write: temp file + rename."""
    path = SOVEREIGN_DIR / filename
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    os.replace(tmp, path)


def load_claims() -> Dict[str, Any]:
    return load_yaml("claims.yaml")


def save_claims(data: Dict[str, Any]) -> None:
    save_yaml("claims.yaml", data)


def load_next_actions() -> Dict[str, Any]:
    return load_yaml("next_actions.json")


def load_evidence_index() -> Dict[str, Any]:
    return load_yaml("evidence/index.json")


def save_evidence_index(data: Dict[str, Any]) -> None:
    save_yaml("evidence/index.json", data)


def load_state() -> Dict[str, Any]:
    with open(SOVEREIGN_DIR / "state.json") as f:
        return json.load(f)


import json
