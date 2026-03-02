# Copyright (C) 2026
#
# Domain-randomized box materials for research datasets.
#
# This module is intentionally lightweight:
# - sampling is deterministic given a seed
# - physics ranges are sourced from `material_definitions.py`
# - visuals are approximated via Principled BSDF parameters (plus optional procedural bump)

from __future__ import annotations

from dataclasses import asdict
from typing import Literal, Optional, Sequence, Tuple

import numpy as np

from infinigen.core.sim.physics import material_definitions as matdefs
from infinigen.core.util.math import int_hash

from .modular_box_factory import BoxMaterialConfig

MaterialFamily = Literal["corrugated", "kraft", "glossy", "plastic"]


def _rgb(r: float, g: float, b: float) -> Tuple[float, float, float]:
    return (float(r), float(g), float(b))


def sample_box_material_config(
    *,
    seed: int,
    allowed_families: Sequence[MaterialFamily] = ("corrugated", "kraft", "glossy", "plastic"),
    variant_id: Optional[int] = None,
) -> BoxMaterialConfig:
    """
    Deterministically sample a domain-randomized box material config.

    Notes:
    - `variant_id` is used to create unique Blender materials via `_deepcopy_<variant_id>`,
      while preserving physics lookup in `material_physics.py`.
    - Physics sampling uses `material_definitions.MATERIALS` ranges.
    """
    if len(allowed_families) == 0:
        raise ValueError("allowed_families must be non-empty")

    rng = np.random.default_rng(int(seed))
    family: MaterialFamily = rng.choice(list(allowed_families))  # type: ignore[assignment]

    # Choose the physics key (material_definitions registry key)
    if family == "kraft":
        physics_key = "cardboard_kraft"
    elif family == "corrugated":
        # Sample a plausible flute subtype deterministically.
        flute = rng.choice(["corrugated_a", "corrugated_b", "corrugated_c", "corrugated_e"])
        physics_key = str(flute)
    elif family == "plastic":
        plastic = rng.choice(["plastic_pp", "plastic_pet", "plastic_hdpe"])
        physics_key = str(plastic)
    elif family == "glossy":
        # Glossy laminated paperboard: physics close to cardboard_white, visuals via clearcoat/roughness.
        physics_key = "cardboard_white"
    else:
        raise ValueError(f"Unhandled family: {family}")

    # Sample physics (deterministic!)
    #
    # IMPORTANT:
    # - Do NOT call `material_definitions.*.sample_parameters()` here because many implementations use
    #   global `np.random`, which would break determinism across repeated calls within one process.
    # - Instead, sample from the min/max ranges using the local RNG.
    try:
        mat = matdefs.get_box_material(str(physics_key))
        density = float(rng.uniform(float(mat.min_density), float(mat.max_density)))
        friction = float(rng.uniform(float(mat.min_friction), float(mat.max_friction)))
        min_r = float(getattr(mat, "min_restitution", 0.1))
        max_r = float(getattr(mat, "max_restitution", min_r))
        restitution = float(rng.uniform(min_r, max_r))
    except Exception:
        # Conservative fallback (should rarely trigger for the allowed box-material keys).
        density = 300.0
        friction = 0.5
        restitution = 0.1

    # Sample visuals
    if family in ("kraft", "corrugated"):
        # Brown paper palette
        base = _rgb(
            rng.uniform(0.50, 0.72),
            rng.uniform(0.40, 0.62),
            rng.uniform(0.24, 0.46),
        )
        roughness = float(rng.uniform(0.72, 0.95))
        specular = float(rng.uniform(0.35, 0.55))
        bump_strength = float(rng.uniform(0.08, 0.30 if family == "corrugated" else 0.18))
        bump_scale = float(rng.uniform(18.0, 70.0))
        clearcoat = 0.0
        transmission = 0.0
    elif family == "glossy":
        # White coated paperboard with lamination
        white = float(rng.uniform(0.78, 0.95))
        base = _rgb(white, white, white)
        roughness = float(rng.uniform(0.10, 0.35))
        specular = float(rng.uniform(0.45, 0.70))
        bump_strength = float(rng.uniform(0.0, 0.06))
        bump_scale = float(rng.uniform(25.0, 80.0))
        clearcoat = float(rng.uniform(0.35, 0.95))
        transmission = 0.0
    elif family == "plastic":
        base = _rgb(
            rng.uniform(0.05, 0.95),
            rng.uniform(0.05, 0.95),
            rng.uniform(0.05, 0.95),
        )
        roughness = float(rng.uniform(0.05, 0.35))
        specular = float(rng.uniform(0.35, 0.70))
        bump_strength = float(rng.uniform(0.0, 0.03))
        bump_scale = float(rng.uniform(20.0, 90.0))
        clearcoat = float(rng.uniform(0.0, 0.35))
        transmission = float(rng.uniform(0.0, 0.35))
    else:
        raise ValueError(f"Unhandled family: {family}")

    # Use a stable variant_id if not provided.
    if variant_id is None:
        variant_id = int_hash((int(seed), str(family), physics_key))

    # The `material_type` should be the physics key so downstream can infer defaults.
    # (material_physics.py strips `_deepcopy_...` during lookup)
    cfg = BoxMaterialConfig(
        material_type=str(physics_key),
        density=density,
        friction=friction,
        restitution=restitution,
        color=base,
        roughness=roughness,
        specular=specular,
        metallic=0.0,
        transmission=transmission,
        ior=1.45,
        clearcoat=clearcoat,
        clearcoat_roughness=0.03,
        bump_strength=bump_strength,
        bump_scale=bump_scale,
        variant_id=int(variant_id),
    )
    return cfg


def box_material_config_to_metadata(cfg: BoxMaterialConfig) -> dict:
    """Convert BoxMaterialConfig to JSON-serializable metadata."""
    d = asdict(cfg)
    # Ensure tuples become lists for JSON friendliness
    if "color" in d and isinstance(d["color"], tuple):
        d["color"] = list(d["color"])
    return d

