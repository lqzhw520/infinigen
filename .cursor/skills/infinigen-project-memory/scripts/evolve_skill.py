#!/usr/bin/env python3
"""Analyze evolution.json trends and auto-update analysis dimensions in rules.json.

This is the self-evolution engine. It reads historical iteration data, detects
patterns not covered by current rules, and updates rules.json + STATUS.md
template accordingly.

Usage:
    python .cursor/skills/infinigen-project-memory/scripts/evolve_skill.py [--project-root .]
"""
import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


def _classify_bug(bug_text: str, existing_categories: list[str]) -> str | None:
    """Try to classify a bug into existing categories. Return None if no match."""
    text = bug_text.lower()
    keyword_map = {
        "URDF_structure": ["urdf", "inertial", "link", "joint", "collision", "material"],
        "rendering_pipeline": ["render", "image", "png", "rgb", "segmentation", "blender", "camera"],
        "physics_simulation": ["pybullet", "physics", "inertia", "self-collision", "dynamics"],
        "data_generation": ["batch", "pipeline", "export", "sample", "dataset", "counter"],
        "dependency_compatibility": ["imageio", "pil", "import", "module", "version", "install"],
    }
    for cat in existing_categories:
        keywords = keyword_map.get(cat, [])
        if any(kw in text for kw in keywords):
            return cat
    return None


def analyze_bugs(iterations: list, rules: dict) -> list[str]:
    """Find bug categories not covered by current rules."""
    existing = set(rules.get("bug_categories", []))
    uncovered = []
    for it in iterations:
        for bug in it.get("bugs_fixed", []):
            bug_text = f"{bug.get('bug', '')} {bug.get('root_cause', '')} {bug.get('fix', '')}"
            cat = _classify_bug(bug_text, list(existing))
            if cat is None:
                uncovered.append(bug_text[:80])
    return uncovered


def analyze_artifacts(iterations: list, rules: dict) -> list[str]:
    """Find artifact types not covered by current rules."""
    existing = set(rules.get("artifact_types", []))
    new_types = set()
    type_keywords = {
        "urdf": ["urdf", "urdf_gt"],
        "dataset": ["dataset", "1k_dataset", "phase1_1k"],
        "model_checkpoint": ["checkpoint", "model", "weights"],
        "documentation": ["doc", "overview", "report", "plan"],
        "proof_image": ["proof", "png", "image", "material_diversity"],
        "script": ["script", "pipeline", "export"],
        "skeleton": ["skeleton", "scaffold"],
    }
    for it in iterations:
        for key, path in it.get("artifacts", {}).items():
            matched = False
            for atype, keywords in type_keywords.items():
                if any(kw in key.lower() or kw in path.lower() for kw in keywords):
                    if atype not in existing:
                        new_types.add(atype)
                    matched = True
                    break
            if not matched:
                new_types.add(f"unknown_{key}")
    return list(new_types)


def analyze_architecture_drift(iterations: list) -> dict:
    """Check if architecture changes span multiple iterations."""
    arch_changes = []
    for it in iterations:
        changes = it.get("architecture_changes", [])
        if changes:
            arch_changes.append({
                "id": it["id"],
                "date": it["date"],
                "changes": changes,
            })
    consecutive = 0
    for i in range(1, len(arch_changes)):
        if arch_changes[i]["id"] - arch_changes[i-1]["id"] <= 2:
            consecutive += 1
    return {
        "total_iterations_with_arch_changes": len(arch_changes),
        "consecutive_arch_change_pairs": consecutive,
        "needs_arch_doc_update": consecutive >= 2,
        "details": arch_changes,
    }


def analyze_lesson_themes(iterations: list) -> dict:
    """Group lessons by theme to detect recurring patterns."""
    all_lessons = []
    for it in iterations:
        all_lessons.extend(it.get("lessons", []))

    themes = Counter()
    theme_keywords = {
        "verification": ["verify", "test", "check", "validate", "pybullet"],
        "compatibility": ["compatibility", "version", "import", "dependency", "imageio", "pil"],
        "urdf_fidelity": ["urdf", "collision", "inertial", "material", "joint", "link"],
        "pipeline_robustness": ["parallel", "batch", "counter", "overwrite", "crash"],
        "determinism": ["rng", "random", "deterministic", "seed", "reproducible"],
    }
    for lesson in all_lessons:
        lesson_lower = lesson.lower()
        matched = False
        for theme, keywords in theme_keywords.items():
            if any(kw in lesson_lower for kw in keywords):
                themes[theme] += 1
                matched = True
                break
        if not matched:
            themes["other"] += 1

    return dict(themes)


def evolve_rules(rules: dict, iterations: list) -> tuple[dict, list[str]]:
    """Update rules based on analysis. Returns (updated_rules, list of changes)."""
    changes = []

    if not rules.get("auto_detect_new_categories", True):
        return rules, ["Auto-detect disabled, no changes made"]

    n = rules.get("evolve_every_n_iterations", 3)
    if len(iterations) % n != 0 and len(iterations) > 0:
        pass

    uncovered_bugs = analyze_bugs(iterations, rules)
    if uncovered_bugs:
        for bug_desc in uncovered_bugs[:3]:
            new_cat = _infer_category_name(bug_desc)
            if new_cat and new_cat not in rules.get("bug_categories", []):
                rules.setdefault("bug_categories", []).append(new_cat)
                changes.append(f"Added bug category: {new_cat} (from: {bug_desc[:40]})")

    new_artifact_types = analyze_artifacts(iterations, rules)
    for at in new_artifact_types:
        if not at.startswith("unknown_"):
            rules.setdefault("artifact_types", []).append(at)
            changes.append(f"Added artifact type: {at}")

    arch = analyze_architecture_drift(iterations)
    if arch["needs_arch_doc_update"]:
        changes.append(
            f"NOTICE: Architecture changes span {arch['total_iterations_with_arch_changes']} iterations "
            f"with {arch['consecutive_arch_change_pairs']} consecutive pairs. "
            f"Consider updating docs/Phase_Architecture_Overview.md"
        )

    themes = analyze_lesson_themes(iterations)
    top_themes = sorted(themes.items(), key=lambda x: -x[1])
    if top_themes:
        current_dims = set(rules.get("analysis_dimensions", []))
        for theme, count in top_themes:
            if count >= 3 and theme not in current_dims and theme != "other":
                rules.setdefault("analysis_dimensions", []).append(theme)
                changes.append(f"Added analysis dimension: {theme} (appeared {count} times in lessons)")

    rules.setdefault("evolution_history", []).append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "version": rules.get("version", 0) + 1,
        "changes": "; ".join(changes) if changes else "No changes needed",
        "trigger": "evolve_skill.py",
        "iterations_analyzed": len(iterations),
        "stats": {
            "total_bugs": sum(len(it.get("bugs_fixed", [])) for it in iterations),
            "total_lessons": sum(len(it.get("lessons", [])) for it in iterations),
            "total_arch_changes": sum(len(it.get("architecture_changes", [])) for it in iterations),
            "lesson_themes": themes,
        },
    })
    rules["version"] = rules.get("version", 0) + 1

    return rules, changes


def _infer_category_name(bug_desc: str) -> str | None:
    """Infer a category name from a bug description."""
    text = bug_desc.lower()
    if any(w in text for w in ["mesh", "obj", "geometry", "vertex"]):
        return "mesh_geometry"
    if any(w in text for w in ["memory", "oom", "leak"]):
        return "memory_management"
    if any(w in text for w in ["network", "api", "http", "connection"]):
        return "network_io"
    if any(w in text for w in ["config", "param", "argument", "flag"]):
        return "configuration"
    return None


def _should_evolve(iterations: list, rules: dict, force: bool) -> bool:
    """Determine if evolution should run based on iteration count and change significance."""
    if force:
        return True
    n = rules.get("evolve_every_n_iterations", 3)
    if len(iterations) > 0 and len(iterations) % n == 0:
        return True
    last_evolve = rules.get("evolution_history", [])
    if not last_evolve:
        return True
    last = last_evolve[-1]
    last_analyzed = last.get("iterations_analyzed", 0)
    if len(iterations) - last_analyzed >= n:
        return True
    total_new_bugs = sum(len(it.get("bugs_fixed", [])) for it in iterations[last_analyzed:])
    total_new_arch = sum(len(it.get("architecture_changes", [])) for it in iterations[last_analyzed:])
    if total_new_bugs >= 2 or total_new_arch >= 2:
        return True
    return False


def _archive_evolution_snapshot(pm_dir: Path, old_version: int, new_version: int,
                                 changes: list, themes: dict, arch: dict):
    """Write a snapshot of this evolution event to history/ for traceability."""
    history_dir = pm_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    path = history_dir / f"{date_str}_skill-evolution-v{old_version}-to-v{new_version}.md"
    lines = [
        f"# Skill Self-Evolution: v{old_version} -> v{new_version}",
        f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Trigger**: evolve_skill.py (auto or explicit)\n",
        "## Changes\n",
    ]
    if changes:
        for c in changes:
            lines.append(f"- {c}")
    else:
        lines.append("- No rule changes needed")
    lines.append("\n## Lesson Theme Distribution\n")
    lines.append("| Theme | Count |")
    lines.append("|-------|-------|")
    for theme, count in sorted(themes.items(), key=lambda x: -x[1]):
        lines.append(f"| {theme} | {count} |")
    lines.append(f"\n## Architecture Drift\n")
    lines.append(f"- Iterations with arch changes: {arch['total_iterations_with_arch_changes']}")
    lines.append(f"- Consecutive pairs: {arch['consecutive_arch_change_pairs']}")
    lines.append(f"- Needs doc update: {'YES' if arch['needs_arch_doc_update'] else 'no'}")
    path.write_text("\n".join(lines))
    return path


def main():
    parser = argparse.ArgumentParser(description="Self-evolve the project memory skill")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--force", action="store_true", help="Force evolution regardless of iteration count")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    pm_dir = root / ".project-memory"
    evo_path = pm_dir / "evolution.json"
    rules_path = pm_dir / "rules.json"

    if not evo_path.exists():
        print(f"ERROR: {evo_path} not found.")
        return 1

    evo = json.loads(evo_path.read_text())
    iterations = evo.get("iterations", [])

    if rules_path.exists():
        rules = json.loads(rules_path.read_text())
    else:
        rules = {
            "version": 0,
            "analysis_dimensions": ["bugs", "architecture", "artifacts", "lessons"],
            "bug_categories": [],
            "artifact_types": [],
            "status_sections": ["Architecture", "Current State", "Milestones", "Bugs", "Artifacts", "Next Steps"],
            "evolve_every_n_iterations": 3,
            "auto_detect_new_categories": True,
            "evolution_history": [],
        }

    if not _should_evolve(iterations, rules, args.force):
        print(f"Skipping evolution: not enough new iterations since last run. Use --force to override.")
        return 0

    print(f"Analyzing {len(iterations)} iterations...")
    old_version = rules.get("version", 0)
    print(f"Current rules version: {old_version}")
    print()

    updated_rules, changes = evolve_rules(rules, iterations)
    new_version = updated_rules.get("version", old_version + 1)

    rules_path.write_text(json.dumps(updated_rules, indent=2, ensure_ascii=False))

    print("=" * 60)
    print("EVOLUTION REPORT")
    print("=" * 60)
    if changes:
        for c in changes:
            print(f"  [+] {c}")
    else:
        print("  No changes needed at this time.")
    print()

    themes = analyze_lesson_themes(iterations)
    print("Lesson theme distribution:")
    for theme, count in sorted(themes.items(), key=lambda x: -x[1]):
        print(f"  {theme}: {count}")
    print()

    arch = analyze_architecture_drift(iterations)
    print(f"Architecture drift: {arch['total_iterations_with_arch_changes']} iterations had changes, "
          f"{arch['consecutive_arch_change_pairs']} consecutive pairs")
    if arch["needs_arch_doc_update"]:
        print("  >>> Architecture documentation may need updating!")
    print()

    snapshot_path = _archive_evolution_snapshot(pm_dir, old_version, new_version, changes, themes, arch)
    print(f"Evolution snapshot: {snapshot_path}")
    print(f"Rules updated: {rules_path}")
    print(f"New version: {new_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
