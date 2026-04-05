#!/usr/bin/env python3
"""
fix_drawerbox_mtl.py — P0b Fix
Patch all empty MTL files in drawerbox URDF assets with cardboard material definitions.

Root cause: sim_exports/urdf/drawerbox/*/assets/*.mtl are empty (only Blender
header comments, no Kd/Ka/Ks definitions). PyBullet's ER_TINY_RENDERER renders
meshes with no material definition as pure white (1.0, 1.0, 1.0), causing
near-white images in all rollout data.

Fix: Write cardboard-accurate MTL material definitions.
Cardboard color sampled from BoxMaterialConfig defaults in modular_box_factory.py:
  cardboard_kraft color = (0.50–0.70, 0.40–0.60, 0.25–0.45) → midpoint ≈ (0.60, 0.50, 0.40)

Usage:
  python scripts/mint/fix_drawerbox_mtl.py [--dry-run]
"""

import argparse
from pathlib import Path

CAMPAIGN_ROOT = Path(__file__).parent.parent.parent.resolve()
DRAWERBOX_ROOT = CAMPAIGN_ROOT / "sim_exports/urdf/drawerbox"

# Cardboard mid-tone (from modular_box_factory.py BoxMaterialConfig defaults)
MTL_TEMPLATE = """# Blender 4.2.0 MTL File: 'None'
# www.blender.org
# P0b fix: Added cardboard material for PyBullet ER_TINY_RENDERER
# Source: BoxMaterialConfig.color = (0.6, 0.5, 0.4) from modular_box_factory.py
# Applies to: drawerbox mesh variant {variant_id}

newmtl cardboard
Ka 0.18 0.15 0.12
Kd 0.60 0.50 0.40
Ks 0.02 0.02 0.02
Ns 10.0
d 1.0
illum 2
"""


def fix_mtl(mtl_path: Path, dry_run: bool = False) -> bool:
    """Patch a single MTL file. Returns True if changed."""
    content = mtl_path.read_text()
    
    # Check if already has material definition (has non-comment, non-blank lines)
    non_comment_lines = [
        line for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if non_comment_lines:
        has_material = any(
            line.startswith("newmtl") or line.startswith("Kd ")
            for line in non_comment_lines
        )
        if has_material:
            print(f"  SKIP (has material): {mtl_path}")
            return False
        # Has content but no material — overwrite with cardboard
        print(f"  OVERWRITE (blank material): {mtl_path}")
    else:
        print(f"  FIX (empty): {mtl_path}")
    
    # Extract variant from path: .../drawerbox/{N}/assets/{name}.mtl
    variant = mtl_path.parent.parent.name
    new_content = MTL_TEMPLATE.format(variant_id=variant)
    
    if not dry_run:
        mtl_path.write_text(new_content)
    
    return not dry_run


def main():
    parser = argparse.ArgumentParser(description="Fix empty MTL files in drawerbox URDF assets")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without modifying files")
    args = parser.parse_args()

    dry_run = args.dry_run
    mode = "DRY-RUN" if dry_run else "LIVE"
    print(f"=== fix_drawerbox_mtl.py [{mode}] ===")
    print(f"  drawerbox root: {DRAWERBOX_ROOT}")
    print()

    mtl_files = sorted(DRAWERBOX_ROOT.glob("*/assets/*.mtl"))
    print(f"  Found {len(mtl_files)} MTL files")

    if dry_run:
        print("  (no files will be modified)")
    print()

    changed = 0
    skipped = 0
    for mtl_path in mtl_files:
        ok = fix_mtl(mtl_path, dry_run=dry_run)
        if ok:
            changed += 1
        else:
            skipped += 1

    print()
    print(f"  Total: {len(mtl_files)}")
    print(f"  Changed: {changed}")
    print(f"  Skipped: {skipped}")

    if dry_run:
        print()
        print("  Re-run WITHOUT --dry-run to apply changes.")
    else:
        print()
        print(f"  ✓ {changed} MTL files patched with cardboard material")
        print("  Note: Images in existing rollouts are baked — new rollouts will have proper colors.")

    # Also check: do we need to patch the _load_drawer function for runtime environments?
    env_file = CAMPAIGN_ROOT / "scripts/mint/drawer_robot_env_lerobot.py"
    if env_file.exists():
        content = env_file.read_text()
        has_change_visual = "changeVisualShape" in content
        print()
        if not has_change_visual:
            print(f"  INFO: {env_file.name} does NOT call changeVisualShape")
            print("  This is OK — ER_TINY_RENDERER reads MTL files directly.")
            print("  If MTL fix doesn't work at runtime, we can add changeVisualShape as a fallback.")
        else:
            print(f"  INFO: {env_file.name} already calls changeVisualShape")


if __name__ == "__main__":
    raise SystemExit(main())
