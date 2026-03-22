# Phase 2 cheap ablations — overnight run

- **Driver**: `overnight_phase2_ablations.sh` (export 6 groups → guided → `evaluate_overnight_ablations.py`)
- **Logs**: `output.log` (tee from driver), `nohup.log` (nohup wrapper), `failures.txt` if any guided step fails
- **Conditioning**: `experiments/physnap/_overnight_sandbox/conditioning/`
- **Guided outputs**: `external/physnap/log/overnight_box_cond_v2__*` (PhysNAP **source** untouched)
- **Table after eval**: `experiments/physnap/_overnight_sandbox/evaluation/overnight_ablations_report.md`

When finished, fill **RESULT.md** here with GPU time and headline comparison vs anchor.
