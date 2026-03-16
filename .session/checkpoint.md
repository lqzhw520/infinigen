# Session Checkpoint
<!-- Auto-synced from .project-memory/STATUS.md by infinigen-project-memory skill -->

**Date**: 2026-03-09
**Branch**: feature/3d-assets
**Last Commit**: d6c74c0a PhysNAP integration: CAPNet rollback + academic review + data bridge pipeline

## Status

See `.project-memory/STATUS.md` for the full project status.

## Completed
- CAPNet rollback: deleted 9 files (scripts, docs, annotations), cleaned STATUS/evolution
- PhysNAP academic review: deep paper analysis with Infinigen integration assessment
- PhysNAP environment: conda physnap (Python 3.9, PyTorch 2.0+cu118, A800 GPU verified)
- Data bridge: infinigen_to_nap.py converts URDF+OBJ to NAP graph format
- 750/1000 samples converted to NAP format (3 types; TuckEnd excluded, K>8)
- NAP compact_pack verification: all converted data passes format check
- Combined dataset: 600 train / 75 val / 75 test across 3 box categories
- Training pipeline: train_physnap_infinigen.sh ready for Option A/B/C

## Next
- [P0-BLOCKED] Download NAP pretrained data from Google Drive
- [P0-BLOCKED] Train PhysNAP baseline on PartNet-Mobility
- [P0-BLOCKED] Retrain PhysNAP with Infinigen data (Option A/B/C)
- [P1] Encode Infinigen shapes via NAP pretrained AE
- [P1] Scale to 10K+ samples
- [P2] Phase 2.1: Topo-Box-Net training

## Lessons
- NAP uses max K=8 nodes -- TuckEndBox (9 parts) exceeds this and needs K=10 or exclusion
- Google Drive is unreachable from AFS cluster -- need manual download or proxy
