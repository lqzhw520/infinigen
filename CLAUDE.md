# CLAUDE.md — Infinigen x PhysNAP Research Project

## Project Overview

This is an academic research project extending [Infinigen](https://github.com/princeton-vl/infinigen) with [PhysNAP](https://github.com/JiahuiLei/NAP.git) for physics-aware neural articulated parts. Current research focus: box conditioning experiments for guided shape generation.

## Environment

- **Server**: A800 dev machine
- **SSH**: `ssh -p 30017 root@10.210.0.88`
- **GPU**: 1x NVIDIA A800-SXM4-80GB (81920 MiB)
- **CPU**: 12-core Intel Xeon 6348 @ 2.6GHz
- **RAM**: 120GB
- **OS**: Ubuntu 20.04 (container)
- **Python**: 3.11.14 (conda env `infinigen`)
- **Conda**: `/root/anaconda3/`
- **Conda activate**: `source /root/anaconda3/etc/profile.d/conda.sh && conda activate infinigen`

## Project Structure

```
/mnt/afs2/zhuhaowu/infinigen/
├── infinigen/              # Infinigen core (submodule)
├── external/physnap/       # PhysNAP (submodule: NAP)
│   ├── core/               # Training code
│   ├── eval/               # Evaluation scripts (run_guided.py, eval.py)
│   ├── configs/nap/        # YAML configs
│   ├── data/               # Datasets
│   └── log/                # Experiment logs & checkpoints
├── experiments/physnap/    # Experiment campaigns
│   ├── box_conditioning_v2/ # Phase 2: conditioning design
│   └── box_prior_v1/       # Phase 1: diagnostics
├── scripts/                # Experiment launcher scripts
├── src/                    # (currently empty)
├── task_plan.md            # Automated task tracker
├── findings.md             # Experiment findings
└── progress.md             # Progress log
```

## Current Research Campaigns

### box_conditioning_v2 (Phase 2: Conditioning Design)
- **Status**: `await_phase2_execution`
- **Base experiment**: `mixed_finetune_k10`
- **Base checkpoint**: `external/physnap/log/v6.1_diffusion_mixed_finetune_infinigen_k10/checkpoint/425_latest.pt`
- **Groups**: zero_singleview, zero_multiview, multistate_singleview, multistate_multiview
- **Hypothesis**: multi-state, multi-view observations provide stronger conditioning signal

### box_prior_v1 (Phase 1: Diagnostics)
- **Status**: `phase1_diagnostics_incomplete`
- **Five-arm matrix**: baseline, scratch_k10_fixed, true_finetune_k10_fixed, mixed_finetune_k10, partial_finetune_k10

## Git

- **Current branch**: `feature/3d-assets`
- **Upstream**: `origin` = `https://github.com/princeton-vl/infinigen.git`
- **Fork**: `my-origin` = `git@github.com:lqzhw520/infinigen.git`
- **Git binary**: `/root/anaconda3/envs/infinigen/bin/git` (only available after conda activate)

## Experiment Commands

All Python commands must be run with conda env activated:
```bash
source /root/anaconda3/etc/profile.d/conda.sh && conda activate infinigen
cd /mnt/afs2/zhuhaowu/infinigen
```

### Run guided conditioning:
```bash
python external/physnap/eval/run_guided.py \
  --name <exp_name> \
  --config_path external/physnap/configs/nap/<config>.yaml \
  --checkpoint_path external/physnap/log/<ckpt>/checkpoint/<pt> \
  --cond_dir experiments/physnap/<campaign>/conditioning/<group> \
  --N 3 --bs 24
```

### Launch with screen (for long jobs):
```bash
screen -dmS <name> bash -c 'source /root/anaconda3/etc/profile.d/conda.sh && conda activate infinigen && cd /mnt/afs2/zhuhaowu/infinigen && python <script> <args> 2>&1 | tee <log>'
```

## Key Packages

- PyTorch 2.3.1 (CUDA)
- matplotlib, numpy, scipy, tensorboard
- wandb: NOT installed (use tensorboard for now)

## Research Skills

ARIS research skills are located at:
- `/mnt/afs2/zhuhaowu/aris-workspace/skills/` (ARIS core + infinigen custom)

Custom infinigen skills:
- `infinigen-experiment` — run experiments directly on A800 (no rsync)

## Safety Rules

1. **NEVER kill existing Python processes** — always check `ps aux | grep python` first
2. **GPU memory** — check `nvidia-smi` before launching; need > 10GB free
3. **Screen sessions** — always use screen for jobs > 5 minutes
4. **State files** — update `task_plan.md`, `findings.md`, `progress.md` after actions
