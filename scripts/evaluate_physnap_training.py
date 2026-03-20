#!/usr/bin/env python3
"""Campaign-grade evaluator for the Infinigen x PhysNAP box-prior study."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import trimesh
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHYSNAP_ROOT = PROJECT_ROOT / "external" / "physnap"


def ensure_binary_compat_preload() -> None:
    conda_prefix = Path(os.environ.get("CONDA_PREFIX", "/root/anaconda3/envs/physnap"))
    libstdcpp = conda_prefix / "lib" / "libstdc++.so.6"
    marker = "CODEX_LIBSTDCPP_PRELOADED"
    if not libstdcpp.exists() or os.environ.get(marker) == "1":
        return
    current = os.environ.get("LD_PRELOAD", "")
    parts = [str(libstdcpp)] + ([current] if current else [])
    os.environ["LD_PRELOAD"] = ":".join(dict.fromkeys(parts))
    os.environ[marker] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], os.environ.copy())


ensure_binary_compat_preload()

CAMPAIGN_DIR = PROJECT_ROOT / "experiments" / "physnap" / "box_prior_v1"
PHYSNAP_PYTHON = Path("/root/anaconda3/envs/physnap/bin/python")
PARTNET_MESH_ROOT = PHYSNAP_ROOT / "data" / "partnet_mobility_graph_mesh"
DEFAULT_BOX_REF_LIMIT = 32
DEFAULT_PARTNET_REF_LIMIT = 64
EVALUATION_STAGES = [
    "export_box_ref",
    "export_partnet_ref",
    "generate_baseline",
    "generate_scratch",
    "generate_finetune",
    "generate_mixed_finetune",
    "generate_partial_finetune",
    "compute_box_metrics",
    "compute_partnet_metrics",
    "audit_conversion",
    "write_report",
]


def ensure_runtime_ld_library_path() -> None:
    conda_prefix = Path(os.environ.get("CONDA_PREFIX", PHYSNAP_PYTHON.parent.parent))
    torch_lib = (
        conda_prefix
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "torch"
        / "lib"
    )
    candidates = [conda_prefix / "lib", torch_lib]
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    prefixes = [str(path) for path in candidates if path.exists()]
    merged = ":".join(prefixes + ([existing] if existing else []))
    if merged:
        os.environ["LD_LIBRARY_PATH"] = merged


ensure_runtime_ld_library_path()


@dataclass
class ReferenceSpec:
    key: str
    name: str
    kind: str
    split: str
    config_path: Path


@dataclass
class ExperimentSpec:
    key: str
    label: str
    config_path: Path
    checkpoint_path: Path
    log_dir: Path
    output_name: str
    evaluation_stage: str


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def progress_file(campaign_dir: Path) -> Path:
    return campaign_dir / "evaluation" / "progress.json"


def init_progress(campaign_dir: Path) -> dict:
    progress = {
        "campaign_id": campaign_dir.name,
        "updated_at": now_iso(),
        "status": "running",
        "distance_backend": None,
        "current_stage": None,
        "latest_artifact": None,
        "message": "evaluation started",
        "stages": {
            stage: {
                "status": "pending",
                "started_at": None,
                "updated_at": None,
                "artifacts": [],
                "message": "",
            }
            for stage in EVALUATION_STAGES
        },
    }
    save_progress(campaign_dir, progress)
    return progress


def load_progress(campaign_dir: Path) -> dict:
    path = progress_file(campaign_dir)
    if not path.exists():
        return init_progress(campaign_dir)
    return json.loads(path.read_text())


def save_progress(campaign_dir: Path, progress: dict) -> None:
    path = progress_file(campaign_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    progress["updated_at"] = now_iso()
    path.write_text(json.dumps(progress, indent=2, ensure_ascii=False))


def update_progress(
    campaign_dir: Path,
    progress: dict,
    stage: str,
    *,
    status: str,
    message: str,
    artifact: str | None = None,
    distance_backend: str | None = None,
) -> None:
    stage_state = progress["stages"][stage]
    if status == "running" and not stage_state["started_at"]:
        stage_state["started_at"] = now_iso()
    stage_state["status"] = status
    stage_state["updated_at"] = now_iso()
    stage_state["message"] = message
    if artifact and artifact not in stage_state["artifacts"]:
        stage_state["artifacts"].append(artifact)
        progress["latest_artifact"] = artifact
    progress["status"] = status if status == "failed" or (stage == "write_report" and status == "completed") else "running"
    progress["current_stage"] = stage
    progress["message"] = message
    if distance_backend:
        progress["distance_backend"] = distance_backend
    save_progress(campaign_dir, progress)


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def add_physnap_imports() -> None:
    for candidate in [str(PHYSNAP_ROOT), str(PHYSNAP_ROOT / "eval")]:
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


def load_manifest(path: Path) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle) or {}


def ensure_physnap_cwd() -> None:
    os.chdir(PHYSNAP_ROOT / "eval")


def detect_distance_backend() -> str:
    add_physnap_imports()
    from eval.instantiation_distance import _pytorch3d_chamfer_distance

    return "pytorch3d" if _pytorch3d_chamfer_distance is not None else "torch_fallback"


def parse_training_metrics(log_dir: Path) -> dict:
    xls_dir = log_dir / "xls"
    if not xls_dir.exists():
        return {}
    metrics = {}
    for csv_file in sorted(xls_dir.glob("*.csv")):
        try:
            data = np.genfromtxt(
                csv_file,
                delimiter=",",
                skip_header=1,
                usecols=(0, 1),
            )
        except Exception:
            continue
        if getattr(data, "ndim", 0) != 2 or len(data) == 0:
            continue
        metrics[csv_file.stem] = {
            "steps": data[:, 0].tolist(),
            "values": data[:, 1].tolist(),
            "start": float(data[0, 1]),
            "end": float(data[-1, 1]),
            "min": float(data[:, 1].min()),
        }
    return metrics


def latest_gallery(run_dir: Path) -> str | None:
    candidates = sorted(run_dir.glob("res_chnk*.png"))
    if not candidates:
        return None
    return str(candidates[-1])


def has_sampled_pcls(run_dir: Path) -> bool:
    pcl_count = len(list((run_dir / "PCL").glob("*.npz")))
    pcl0_count = len(list((run_dir / "PCL0").glob("*.npz")))
    return pcl_count > 0 and pcl0_count > 0


def resolve_checkpoint(experiment: ExperimentSpec) -> Path:
    if experiment.checkpoint_path.exists():
        return experiment.checkpoint_path
    checkpoint_dir = experiment.log_dir / "checkpoint"
    latest = sorted(checkpoint_dir.glob("*_latest.pt"))
    if latest:
        return latest[-1]
    numbered = sorted(checkpoint_dir.glob("*.pt"))
    if numbered:
        return numbered[-1]
    raise FileNotFoundError(f"No checkpoint found for {experiment.key} in {checkpoint_dir}")


def prepare_dataset_cfg(config_path: Path) -> dict:
    add_physnap_imports()
    from init.config_utils import load_config

    cfg = load_config(str(config_path), default_path=str(PHYSNAP_ROOT / "init" / "default.yaml"))
    cfg["root"] = str(PHYSNAP_ROOT)
    cfg["modes"] = ["train", "val", "test"]
    cfg["dataset"]["dataset_proportion"] = [1.0, 1.0, 1.0]
    for key in ["split_path", "embedding_index_file", "embedding_precompute_path"]:
        value = cfg["dataset"].get(key)
        if value and not os.path.isabs(value):
            cfg["dataset"][key] = str(PHYSNAP_ROOT / value)
    shapeprior = cfg["model"]["part_shape_prior"].get("pretrained_shapeprior_path")
    if shapeprior and not os.path.isabs(shapeprior):
        cfg["model"]["part_shape_prior"]["pretrained_shapeprior_path"] = str(PHYSNAP_ROOT / shapeprior)
    return cfg


def attach_meshes(graph, raw_nodes, mesh_name_list):
    attached = False
    for node_idx, node in enumerate(raw_nodes):
        mesh_data = node.get("agg_mesh")
        if mesh_data is None:
            continue
        vertices, faces = mesh_data
        mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
        graph.nodes[node_idx]["mesh"] = mesh
        attached = True
    if attached:
        return graph

    for node_idx, mesh_name in enumerate(mesh_name_list):
        if not mesh_name:
            continue
        mesh_path = PARTNET_MESH_ROOT / f"{mesh_name}.off"
        if not mesh_path.exists():
            continue
        graph.nodes[node_idx]["mesh"] = trimesh.load(mesh_path, force="mesh", process=False)
    return graph


def reference_outputs_ready(ref_root: Path, reference_name: str, require_id_matrices: bool = True) -> bool:
    required = [
        ref_root / "PCL",
        ref_root / "PCL0",
    ]
    if require_id_matrices:
        required.extend([
            ref_root / "ID_D_matrix" / f"ref={reference_name}_nstates=10_npcl=2048.npz",
            ref_root / "ID_D_matrix" / f"ref={reference_name}_nstates=1_npcl=2048.npz",
        ])
    return all(path.exists() for path in required)


def ensure_reference_subset(reference_root: Path, subset_name: str, limit: int, *, force: bool = False) -> Path:
    add_physnap_imports()
    from eval.instantiation_distance import compute_D_matrix

    subset_root = PHYSNAP_ROOT / "log" / subset_name
    pcl_src = reference_root / "PCL"
    pcl0_src = reference_root / "PCL0"
    pcl_dst = subset_root / "PCL"
    pcl0_dst = subset_root / "PCL0"
    id_dir = subset_root / "ID_D_matrix"

    if not force and reference_outputs_ready(subset_root, subset_name):
        return subset_root

    if force and subset_root.exists():
        for child in sorted(subset_root.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()

    pcl_dst.mkdir(parents=True, exist_ok=True)
    pcl0_dst.mkdir(parents=True, exist_ok=True)

    for src_dir, dst_dir in [(pcl_src, pcl_dst), (pcl0_src, pcl0_dst)]:
        files = sorted(src_dir.glob("*.npz"))[:limit]
        for src in files:
            dst = dst_dir / src.name
            if dst.exists() or dst.is_symlink():
                continue
            dst.symlink_to(src)

    compute_D_matrix(str(pcl_dst), str(pcl_dst), str(id_dir), N_states_max=10, N_pcl_max=2048, ref_name=subset_name)
    compute_D_matrix(str(pcl0_dst), str(pcl0_dst), str(id_dir), N_states_max=1, N_pcl_max=2048, ref_name=subset_name)
    return subset_root


def export_reference(reference: ReferenceSpec, *, force: bool = False, compute_id_matrices: bool = True) -> Path:
    add_physnap_imports()
    from dataset import get_dataset
    from eval.instantiation_distance import compute_D_matrix
    from eval.sample_pcl import sample as sample_graph
    from init import setup_seed
    from object_utils.arti_graph_utils_v3 import get_G_from_VE

    ref_root = PHYSNAP_ROOT / "log" / reference.name
    pcl_dir = ref_root / "PCL"
    pcl0_dir = ref_root / "PCL0"
    id_dir = ref_root / "ID_D_matrix"
    if not force and reference_outputs_ready(ref_root, reference.name, require_id_matrices=compute_id_matrices):
        return ref_root

    if not force and pcl_dir.exists() and pcl0_dir.exists():
        if compute_id_matrices:
            id_dir.mkdir(parents=True, exist_ok=True)
            compute_D_matrix(str(pcl_dir), str(pcl_dir), str(id_dir), N_states_max=10, N_pcl_max=2048, ref_name=reference.name)
            compute_D_matrix(str(pcl0_dir), str(pcl0_dir), str(id_dir), N_states_max=1, N_pcl_max=2048, ref_name=reference.name)
        return ref_root

    cfg = prepare_dataset_cfg(reference.config_path)
    dataset_cls = get_dataset(cfg)
    dataset = dataset_cls(cfg, reference.split)
    data_root = Path(cfg["root"]) / cfg["dataset"]["data_root"]

    g_dir = ref_root / "G"
    g_dir.mkdir(parents=True, exist_ok=True)
    pcl_dir.mkdir(parents=True, exist_ok=True)
    pcl0_dir.mkdir(parents=True, exist_ok=True)

    setup_seed(12345)

    for data, meta in dataset:
        object_id = meta["partnet-m-id"]
        V = (data["V"] * data["V_scale"]).cpu().numpy()
        E = (data["E"] * data["E_scale"]).cpu().numpy()
        graph = get_G_from_VE(V, E)

        raw_npz = np.load(data_root / f"{object_id}.npz", allow_pickle=True)
        raw_nodes = raw_npz["V"].tolist()
        graph = attach_meshes(graph, raw_nodes, meta.get("mesh_name_list", []))

        graph_path = g_dir / f"{object_id}.pkl"
        with graph_path.open("wb") as handle:
            pickle.dump(graph, handle)
        sample_graph(str(graph_path), str(pcl_dir / f"{object_id}.npz"), N_states=10, N_PCL=2048, zero_state=False)
        sample_graph(str(graph_path), str(pcl0_dir / f"{object_id}.npz"), N_states=1, N_PCL=2048, zero_state=True)

    if compute_id_matrices:
        id_dir = ref_root / "ID_D_matrix"
        compute_D_matrix(str(pcl_dir), str(pcl_dir), str(id_dir), N_states_max=10, N_pcl_max=2048, ref_name=reference.name)
        compute_D_matrix(str(pcl0_dir), str(pcl0_dir), str(id_dir), N_states_max=1, N_pcl_max=2048, ref_name=reference.name)
    return ref_root


def run_generation(experiment: ExperimentSpec, reference: ReferenceSpec, *, num_samples: int, batch_size: int, force: bool) -> Path:
    run_dir = PHYSNAP_ROOT / "log" / experiment.output_name
    stats_path = run_dir / "stats.json"
    needs_regeneration = run_dir.exists() and not has_sampled_pcls(run_dir)
    if stats_path.exists() and not force and not needs_regeneration:
        return run_dir

    checkpoint_path = resolve_checkpoint(experiment)
    command = [
        str(PHYSNAP_PYTHON),
        "run_guided.py",
        "--name",
        experiment.key,
        "--output_name",
        experiment.output_name,
        "--config_path",
        str(experiment.config_path),
        "--checkpoint_path",
        str(checkpoint_path),
        "--ref_name",
        reference.name,
        "--ref_path",
        str(PHYSNAP_ROOT / "log" / reference.name),
        "--N",
        str(num_samples),
        "--bs",
        str(batch_size),
    ]
    if (force or needs_regeneration) and run_dir.exists():
        command.append("--remove")
    subprocess.run(command, cwd=PHYSNAP_ROOT / "eval", check=True)
    return run_dir


def compute_metric_bundle(run_dir: Path, run_name: str, ref_root: Path, ref_name: str) -> dict:
    add_physnap_imports()
    from eval.compute_metrics import compute_gen_metrics
    from eval.instantiation_distance import compute_D_matrix

    if not has_sampled_pcls(run_dir):
        raise RuntimeError(f"Generated run `{run_name}` has no sampled PCL/PCL0 artifacts under {run_dir}")

    bundle = {}

    multi_dir = run_dir / f"ID_D_matrix__{ref_name}"
    sr_multi = multi_dir / f"ref={ref_name}_nstates=10_npcl=2048.npz"
    if not sr_multi.exists():
        _, sr_multi = compute_D_matrix(
            str(run_dir / "PCL"),
            str(ref_root / "PCL"),
            str(multi_dir),
            N_states_max=10,
            N_pcl_max=2048,
            ref_name=ref_name,
            return_fn=True,
        )
    ss_multi = multi_dir / f"ref={run_name}_nstates=10_npcl=2048.npz"
    if not ss_multi.exists():
        _, ss_multi = compute_D_matrix(
            str(run_dir / "PCL"),
            str(run_dir / "PCL"),
            str(multi_dir),
            N_states_max=10,
            N_pcl_max=2048,
            ref_name=run_name,
            return_fn=True,
        )
    bundle["multi_state"] = compute_gen_metrics(
        gen=run_name,
        ref=ref_name,
        N_states=10,
        N_pcl=2048,
        retrieval=False,
        sr_fn=sr_multi,
        ss_fn=ss_multi,
        dump_metrics=False,
    )

    zero_dir = run_dir / f"ID_D_matrix0__{ref_name}"
    sr_zero = zero_dir / f"ref={ref_name}_nstates=1_npcl=2048.npz"
    if not sr_zero.exists():
        _, sr_zero = compute_D_matrix(
            str(run_dir / "PCL0"),
            str(ref_root / "PCL0"),
            str(zero_dir),
            N_states_max=1,
            N_pcl_max=2048,
            ref_name=ref_name,
            return_fn=True,
        )
    ss_zero = zero_dir / f"ref={run_name}_nstates=1_npcl=2048.npz"
    if not ss_zero.exists():
        _, ss_zero = compute_D_matrix(
            str(run_dir / "PCL0"),
            str(run_dir / "PCL0"),
            str(zero_dir),
            N_states_max=1,
            N_pcl_max=2048,
            ref_name=run_name,
            return_fn=True,
        )
    bundle["zero_state"] = compute_gen_metrics(
        gen=run_name,
        ref=ref_name,
        N_states=1,
        N_pcl=2048,
        retrieval=False,
        sr_fn=sr_zero,
        ss_fn=ss_zero,
        dump_metrics=False,
    )
    return bundle


def build_markdown(summary: dict) -> str:
    lines = [
        "# PhysNAP Box Prior Evaluation",
        "",
        f"- Generated: {summary['generated_at']}",
        f"- Campaign: {summary['campaign_id']}",
        f"- Box reference export: {summary['box_reference']}",
        f"- Box metric reference: {summary.get('box_metric_reference', summary['box_reference'])}",
        f"- Box metric reference size: {summary.get('box_metric_reference_size', 'unknown')}",
        f"- PartNet sanity reference: {summary.get('partnet_metric_reference', summary['partnet_reference'])}",
        f"- PartNet sanity reference size: {summary.get('partnet_metric_reference_size', 'unknown')}",
        "",
        "## Box-Domain Metrics",
        "",
        "| Experiment | split | MMD | COV | 1NN-acc | E_pen | E_mob |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, payload in summary["experiments"].items():
        box_multi = payload["box_metrics"]["multi_state"]
        physics = payload["physics_stats"]
        lines.append(
            "| {label} | multi | {mmd:.4f} | {cov:.4f} | {acc:.4f} | {pen:.6f} | {mob:.6f} |".format(
                label=payload["label"],
                mmd=box_multi["mmd"],
                cov=box_multi["cov"],
                acc=box_multi["1NN-acc"],
                pen=physics.get("pen_error_mean", float("nan")),
                mob=physics.get("mob_error_mean", float("nan")),
            )
        )
        box_zero = payload["box_metrics"]["zero_state"]
        lines.append(
            "| {label} | zero | {mmd:.4f} | {cov:.4f} | {acc:.4f} | - | - |".format(
                label=payload["label"],
                mmd=box_zero["mmd"],
                cov=box_zero["cov"],
                acc=box_zero["1NN-acc"],
            )
        )
    lines.extend(
        [
            "",
            "## PartNet Sanity",
            "",
            "| Experiment | split | MMD | COV | 1NN-acc |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for payload in summary["experiments"].values():
        sanity = payload["partnet_sanity"]["multi_state"]
        lines.append(
            "| {label} | multi | {mmd:.4f} | {cov:.4f} | {acc:.4f} |".format(
                label=payload["label"],
                mmd=sanity["mmd"],
                cov=sanity["cov"],
                acc=sanity["1NN-acc"],
            )
        )
    diagnostics = summary.get("diagnostics", {})
    if diagnostics:
        gt_eval = diagnostics.get("gt_self_eval", {})
        partnet_gt = diagnostics.get("partnet_gt_self_eval", {})
        saved_fidelity = diagnostics.get("saved_dataset_fidelity", {})
        lines.extend(
            [
                "",
                "## Conversion Audit",
                "",
                f"- Decision: `{diagnostics.get('decision')}`",
                f"- Box GT self-eval pen mean: `{gt_eval.get('pen_error_mean')}`",
                f"- Box GT self-eval mob mean: `{gt_eval.get('mob_error_mean')}`",
                f"- PartNet GT self-eval pen mean: `{partnet_gt.get('pen_error_mean')}`",
                f"- PartNet GT self-eval mob mean: `{partnet_gt.get('mob_error_mean')}`",
                f"- Saved bbox vs fixed converter max error: `{saved_fidelity.get('max_saved_bbox_vs_fixed_converter_error')}`",
                f"- Visual-origin bbox effect max: `{saved_fidelity.get('max_bbox_delta_due_to_visual_origin')}`",
            ]
        )
        for reason in diagnostics.get("blocker_reasons", []):
            lines.append(f"- Blocker: {reason}")
    lines.extend(["", "## Sample Galleries", ""])
    for payload in summary["experiments"].values():
        lines.append(f"- `{payload['label']}`: `{payload.get('gallery') or 'missing'}`")
    return "\n".join(lines) + "\n"


def load_references(manifest: dict) -> tuple[ReferenceSpec, ReferenceSpec]:
    refs = manifest["references"]
    box_ref = ReferenceSpec(
        key="box_val",
        name=refs["box_val"]["name"],
        kind=refs["box_val"]["kind"],
        split=refs["box_val"]["split"],
        config_path=resolve_path(refs["box_val"]["config_path"]),
    )
    partnet_ref = ReferenceSpec(
        key="partnet_val",
        name=refs["partnet_val"]["name"],
        kind=refs["partnet_val"]["kind"],
        split=refs["partnet_val"]["split"],
        config_path=resolve_path(refs["partnet_val"]["config_path"]),
    )
    return box_ref, partnet_ref


def load_experiments(manifest: dict) -> list[ExperimentSpec]:
    experiments = []
    campaign_id = manifest["campaign"]["id"]
    for key, payload in manifest["experiments"].items():
        if payload.get("enabled", True) is False:
            continue
        experiments.append(
            ExperimentSpec(
                key=key,
                label=payload["label"],
                config_path=resolve_path(payload["config_path"]),
                checkpoint_path=resolve_path(payload["checkpoint_path"]),
                log_dir=resolve_path(payload["log_dir"]),
                output_name=f"{campaign_id}__{key}",
                evaluation_stage=payload.get("evaluation_stage", f"generate_{key}"),
            )
        )
    return experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PhysNAP box-prior experiments")
    parser.add_argument("--campaign-dir", default=str(CAMPAIGN_DIR))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--box-ref-limit", type=int, default=DEFAULT_BOX_REF_LIMIT)
    parser.add_argument("--partnet-ref-limit", type=int, default=DEFAULT_PARTNET_REF_LIMIT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    ensure_physnap_cwd()
    args = parse_args()
    campaign_dir = Path(args.campaign_dir).resolve()
    output_dir = campaign_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest).resolve() if args.manifest else campaign_dir / "manifest.yaml"
    manifest = load_manifest(manifest_path)
    box_ref, partnet_ref = load_references(manifest)
    experiments = load_experiments(manifest)
    progress = init_progress(campaign_dir)
    try:
        distance_backend = detect_distance_backend()
        update_progress(
            campaign_dir,
            progress,
            "export_box_ref",
            status="running",
            message=f"Exporting box reference `{box_ref.name}`",
            distance_backend=distance_backend,
        )

        box_ref_root = PHYSNAP_ROOT / "log" / box_ref.name
        box_ref_requires_full_id = not (args.box_ref_limit and args.box_ref_limit > 0)
        if not args.force and (box_ref_root / "PCL").exists() and (box_ref_root / "PCL0").exists():
            update_progress(
                campaign_dir,
                progress,
                "export_box_ref",
                status="running",
                message=f"Reusing existing sampled box reference `{box_ref.name}`",
            )
        else:
            box_ref_root = export_reference(box_ref, force=args.force, compute_id_matrices=box_ref_requires_full_id)
        box_metric_ref_root = box_ref_root
        box_metric_ref_name = box_ref.name
        if args.box_ref_limit and args.box_ref_limit > 0:
            subset_name = f"{box_ref.name}_eval{args.box_ref_limit}"
            update_progress(
                campaign_dir,
                progress,
                "export_box_ref",
                status="running",
                message=f"Building box metric reference subset `{subset_name}`",
            )
            box_metric_ref_root = ensure_reference_subset(
                box_ref_root,
                subset_name,
                args.box_ref_limit,
                force=args.force,
            )
            box_metric_ref_name = box_metric_ref_root.name
        update_progress(
            campaign_dir,
            progress,
            "export_box_ref",
            status="completed",
            message=f"Reference `{box_metric_ref_name}` ready for box metrics",
            artifact=str(box_metric_ref_root),
        )

        update_progress(
            campaign_dir,
            progress,
            "export_partnet_ref",
            status="running",
            message=f"Exporting PartNet sanity reference `{partnet_ref.name}`",
        )
        partnet_ref_requires_full_id = not (args.partnet_ref_limit and args.partnet_ref_limit > 0)
        partnet_ref_root = export_reference(partnet_ref, force=args.force, compute_id_matrices=partnet_ref_requires_full_id)
        partnet_metric_ref_root = partnet_ref_root
        partnet_metric_ref_name = partnet_ref.name
        if args.partnet_ref_limit and args.partnet_ref_limit > 0:
            subset_name = f"{partnet_ref.name}_eval{args.partnet_ref_limit}"
            update_progress(
                campaign_dir,
                progress,
                "export_partnet_ref",
                status="running",
                message=f"Building PartNet sanity subset `{subset_name}`",
            )
            partnet_metric_ref_root = ensure_reference_subset(
                partnet_ref_root,
                subset_name,
                args.partnet_ref_limit,
                force=args.force,
            )
            partnet_metric_ref_name = partnet_metric_ref_root.name
        update_progress(
            campaign_dir,
            progress,
            "export_partnet_ref",
            status="completed",
            message=f"Reference `{partnet_metric_ref_name}` ready",
            artifact=str(partnet_metric_ref_root),
        )

        summary = {
            "campaign_id": manifest["campaign"]["id"],
            "generated_at": now_iso(),
            "box_reference": box_ref.name,
            "box_metric_reference": box_metric_ref_name,
            "box_metric_reference_size": len(list((box_metric_ref_root / "PCL").glob("*.npz"))),
            "partnet_reference": partnet_ref.name,
            "partnet_metric_reference": partnet_metric_ref_name,
            "partnet_metric_reference_size": len(list((partnet_metric_ref_root / "PCL").glob("*.npz"))),
            "experiments": {},
        }

        run_dirs = {}
        physics_stats_by_key = {}
        checkpoints_by_key = {}
        for experiment in experiments:
            stage = experiment.evaluation_stage
            update_progress(
                campaign_dir,
                progress,
                stage,
                status="running",
                message=f"Preparing generation outputs for `{experiment.key}`",
            )
            run_dir = run_generation(
                experiment,
                box_ref,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                force=args.force,
            )
            stats_path = run_dir / "stats.json"
            physics_stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
            run_dirs[experiment.key] = run_dir
            physics_stats_by_key[experiment.key] = physics_stats
            checkpoints_by_key[experiment.key] = str(resolve_checkpoint(experiment))
            update_progress(
                campaign_dir,
                progress,
                stage,
                status="completed",
                message=f"Generation ready for `{experiment.key}`",
                artifact=str(run_dir),
            )

        update_progress(
            campaign_dir,
            progress,
            "compute_box_metrics",
            status="running",
            message="Computing box-domain MMD/COV/1NN metrics",
        )
        for experiment in experiments:
            run_dir = run_dirs[experiment.key]
            physics_stats = physics_stats_by_key[experiment.key]
            summary["experiments"][experiment.key] = {
                "label": experiment.label,
                "run_dir": str(run_dir),
                "config_path": str(experiment.config_path),
                "checkpoint_path": checkpoints_by_key[experiment.key],
                "training_metrics": parse_training_metrics(experiment.log_dir),
                "physics_stats": {
                    "pen_error_mean": physics_stats.get("pen_error_mean"),
                    "mob_error_mean": physics_stats.get("mob_error_mean"),
                    "cond_error_mean": physics_stats.get("cond_error_mean"),
                },
                "box_metrics": compute_metric_bundle(
                    run_dir,
                    experiment.output_name,
                    box_metric_ref_root,
                    box_metric_ref_name,
                ),
                "gallery": latest_gallery(run_dir),
            }
            update_progress(
                campaign_dir,
                progress,
                "compute_box_metrics",
                status="running",
                message=f"Computed box metrics for `{experiment.key}`",
                artifact=str(run_dir / f"ID_D_matrix__{box_metric_ref_name}"),
            )
        update_progress(
            campaign_dir,
            progress,
            "compute_box_metrics",
            status="completed",
            message="Box-domain metrics complete",
        )

        update_progress(
            campaign_dir,
            progress,
            "compute_partnet_metrics",
            status="running",
            message="Computing PartNet sanity metrics",
        )
        for experiment in experiments:
            run_dir = run_dirs[experiment.key]
            summary["experiments"][experiment.key]["partnet_sanity"] = compute_metric_bundle(
                run_dir,
                experiment.output_name,
                partnet_metric_ref_root,
                partnet_metric_ref_name,
            )
            update_progress(
                campaign_dir,
                progress,
                "compute_partnet_metrics",
                status="running",
                message=f"Computed PartNet sanity metrics for `{experiment.key}`",
                artifact=str(run_dir / f"ID_D_matrix__{partnet_metric_ref_name}"),
            )
        update_progress(
            campaign_dir,
            progress,
            "compute_partnet_metrics",
            status="completed",
            message="PartNet sanity metrics complete",
        )

        update_progress(
            campaign_dir,
            progress,
            "audit_conversion",
            status="running",
            message="Running conversion audit and GT self-eval",
        )
        audit_script = resolve_path(manifest["diagnostics"]["conversion_audit"])
        subprocess.run(
            [
                str(PHYSNAP_PYTHON),
                str(audit_script),
                "--campaign-dir",
                str(campaign_dir),
                "--manifest",
                str(manifest_path),
                "--gt-limit",
                str(args.box_ref_limit),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
        audit_path = campaign_dir / "diagnostics" / "conversion_audit.json"
        summary["diagnostics"] = json.loads(audit_path.read_text()) if audit_path.exists() else {}
        update_progress(
            campaign_dir,
            progress,
            "audit_conversion",
            status="completed",
            message="Conversion audit complete",
            artifact=str(audit_path),
        )

        summary_path = output_dir / "comparison_summary.json"
        report_path = output_dir / "comparison_report.md"
        update_progress(
            campaign_dir,
            progress,
            "write_report",
            status="running",
            message="Writing summary and markdown report",
        )
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        report_path.write_text(build_markdown(summary))
        update_progress(
            campaign_dir,
            progress,
            "write_report",
            status="completed",
            message="Evaluation summary and report are ready",
            artifact=str(report_path),
        )
        print(f"Summary saved to: {summary_path}")
        print(f"Report saved to: {report_path}")
        return 0
    except Exception as exc:
        failed_stage = progress.get("current_stage") or "write_report"
        update_progress(
            campaign_dir,
            progress,
            failed_stage,
            status="failed",
            message=f"{type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
