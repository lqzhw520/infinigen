"""
Harness v1.4 — YAML/JSON utilities for sovereign file I/O.
Atomic writes: write to temp file, then rename.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

CAMPAIGN_ROOT = Path(__file__).parent.parent.parent.resolve()
SOVEREIGN_DIR = CAMPAIGN_ROOT / "sovereign"
sys.path.insert(0, str(CAMPAIGN_ROOT / "scripts" / "harness"))


# ── YAML I/O ───────────────────────────────────────────────────────────────────

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


# ── JSON I/O ─────────────────────────────────────────────────────────────────

def load_json(filename: str) -> Dict[str, Any]:
    """Load a sovereign JSON file (atomic read)."""
    path = SOVEREIGN_DIR / filename
    with open(path) as f:
        return json.load(f)


def save_json(filename: str, data: Dict[str, Any]) -> None:
    """Atomic write: temp file + rename. JSON only."""
    path = SOVEREIGN_DIR / filename
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ── Typed loaders/savers ─────────────────────────────────────────────────────

def load_claims() -> Dict[str, Any]:
    return load_yaml("claims.yaml")


def save_claims(data: Dict[str, Any]) -> None:
    save_yaml("claims.yaml", data)


def load_next_actions() -> Dict[str, Any]:
    return load_yaml("next_actions.json")


def load_state() -> Dict[str, Any]:
    with open(SOVEREIGN_DIR / "state.json") as f:
        return json.load(f)


# ── Evidence index: JSON only ────────────────────────────────────────────────
# evidence/index.json is JSON (NOT YAML) — readers expect json.load().
# Writers must use save_evidence_index() which calls save_json().

def load_evidence_index() -> Dict[str, Any]:
    """Load evidence/index.json as JSON."""
    return load_json("evidence/index.json")


def save_evidence_index(data: Dict[str, Any]) -> None:
    """Save evidence/index.json as JSON (atomic)."""
    save_json("evidence/index.json", data)
