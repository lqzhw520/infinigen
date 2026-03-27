# Progress

- 2026-03-20T21:00:00+08:00: Synced final Phase 2 terminal verdict into `.project-memory/history/` with a dedicated `claim_not_supported` snapshot.
- 2026-03-20T21:00:00+08:00: Wrote `experiments/physnap/box_conditioning_v2/root_cause_tree.md` to record the strict four-layer negative-result analysis.
- 2026-03-20T21:00:00+08:00: Repaired the project-memory hook contract so post-commit validation no longer rewrites tracked files and re-dirties the worktree.
- 2026-03-25T19:30:00+08:00: [MINT] Fixed Layer 0 bug — `attachment_local` not set in `rollout_policy()`. Built per-seed grasp cache from C2 AnyGrasp NPZ data.
- 2026-03-25T19:45:00+08:00: [MINT] D1 re-run PASSED — seed_009_ep02::extended_overfit promoted (success_gain=0.2, grasp_gain=0.4, pull_gain=0.368).
- 2026-03-25T19:50:00+08:00: [MINT] Generated oracle grasp cache for all 15 seeds (train + held-out) for D3/E1.
- 2026-03-25T19:58:00+08:00: [MINT] D2 launched — base_overfit(600), extended_overfit(1200), phase_balanced_overfit(1200).
- 2026-03-25T21:34:00+08:00: [MINT] D2 FAILED — all 3 variants below gate. Best: extended_overfit 20% success (1/5), success_gain=0.2 but only 1 source rollout.
- 2026-03-25T21:46:00+08:00: [MINT] D3 launched with soft D2 dependency (positive transfer from extended_overfit).
- 2026-03-25T22:27:00+08:00: [MINT] D3 FAILED — finetuned 0% vs pretrained 33% on train seeds. Catastrophic forgetting confirmed.
- 2026-03-25T22:29:00+08:00: [MINT] E1 launched with soft D3 dependency (training completed).
- 2026-03-25T22:43:00+08:00: [MINT] E1 COMPLETED — verdict=`scientific_not_supported`. Finetuned 0% vs pretrained 20% on held-out seeds. Random policy (20%) ties pretrained.
