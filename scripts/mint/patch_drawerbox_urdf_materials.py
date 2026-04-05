#!/usr/bin/env python3
"""
patch_drawerbox_urdf_materials.py — P0b Fix (Runtime URDF patch)
Adds inline <material> definitions to drawerbox URDF <visual> elements so that
ER_TINY_RENDERER renders cardboard-colored geometry instead of default white.

ER_TINY_RENDERER reads <material> tags from URDF but ignores:
  1. .mtl files (they're empty in drawerbox assets)
  2. changeVisualShape rgbaColor (software renderer limitation)
  3. GEOM_BOX rgbaColor (geometry-only shapes)

The fix: inject <material name="cardboard"><color rgba="0.60 0.50 0.40 1.0"/></material>
into every <visual> block in each drawerbox URDF.

Usage:
  python scripts/mint/patch_drawerbox_urdf_materials.py [--dry-run]
"""

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

CAMPAIGN_ROOT = Path(__file__).parent.parent.parent.resolve()
DRAWERBOX_ROOT = CAMPAIGN_ROOT / "sim_exports/urdf/drawerbox"

# Cardboard color matching modular_box_factory.py BoxMaterialConfig Kd
# Kd = (0.60, 0.50, 0.40) in RGB linear space
MATERIAL_XML = '<material name="cardboard"><color rgba="0.60 0.50 0.40 1.0"/></material>'


def patch_urdf(urdf_path: Path, dry_run: bool = False) -> bool:
    """Add <material> to <visual> blocks that lack one. Returns True if changed."""
    content = urdf_path.read_text()
    
    try:
        tree = ET.parse(str(urdf_path))
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"  ERROR (XML parse failed): {urdf_path} — {e}")
        return False

    # ET namespace handling
    ns = {'urdf': 'http://ros.org/schemas/urdf'}
    
    changed = False
    for visual in root.iter('visual'):
        # Check if already has material with color AND name attribute
        material = visual.find('material')
        color = material.find('color') if material is not None else None
        has_name = material is not None and 'name' in material.attrib
        has_color = color is not None
        
        if has_name and has_color:
            continue  # fully specified, skip
        
        if material is None:
            material = ET.SubElement(visual, 'material')
            material.set('name', 'cardboard')
            color_elem = ET.SubElement(material, 'color')
            color_elem.set('rgba', '0.60 0.50 0.40 1.0')
            changed = True
        else:
            # Material exists — ensure it has name AND color
            if 'name' not in material.attrib:
                material.set('name', 'cardboard')
                changed = True
            color = material.find('color')
            if color is None:
                color_elem = ET.SubElement(material, 'color')
                color_elem.set('rgba', '0.60 0.50 0.40 1.0')
                changed = True
            elif changed:  # name was added, color existed — still counts as changed
                pass

    if not changed:
        print(f"  SKIP (has material): {urdf_path}")
        return False

    if dry_run:
        print(f"  PATCH (would add material): {urdf_path}")
        return True  # would-change, count as changed even in dry-run

    # Write back with XML declaration preserved
    tree_str = ET.tostring(root, encoding='unicode')
    # Add XML declaration
    xml_decl = '<?xml version="1.0" ?>\n'
    urdf_path.write_text(xml_decl + tree_str)
    print(f"  PATCHED: {urdf_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Patch drawerbox URDF materials for ER_TINY_RENDERER")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dry_run = args.dry_run
    mode = "DRY-RUN" if dry_run else "LIVE"
    print(f"=== patch_drawerbox_urdf_materials.py [{mode}] ===")
    print(f"  drawerbox root: {DRAWERBOX_ROOT}")
    print()

    urdf_files = sorted(DRAWERBOX_ROOT.glob("*/drawerbox.urdf"))
    print(f"  Found {len(urdf_files)} URDF files")
    if dry_run:
        print("  (no files will be modified)")
    print()

    changed = 0
    skipped = 0
    for urdf_path in urdf_files:
        result = patch_urdf(urdf_path, dry_run=dry_run)
        if result:
            changed += 1
        else:
            skipped += 1

    print()
    print(f"  Total: {len(urdf_files)}")
    print(f"  Changed: {changed}")
    print(f"  Skipped: {skipped}")

    if dry_run:
        print()
        print("  Re-run WITHOUT --dry-run to apply changes.")
    else:
        print()
        print(f"  ✓ {changed} URDF files patched with cardboard material")
        print("  P0b fix complete — re-run rollout recording to capture colored images.")


if __name__ == "__main__":
    raise SystemExit(main())
