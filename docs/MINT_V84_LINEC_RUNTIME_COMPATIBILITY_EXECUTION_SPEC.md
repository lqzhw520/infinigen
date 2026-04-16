Current authoritative status:
- Wrapper adapter for top-level vs legacy PaliGemma image-feature access is implemented.
- `scripts/mint/run_g8_runtime_compat_smoke.py` now exists and runs.
- The authoritative smoke now reaches `image_features_resolved`; the earlier `.model` access failure and first vision-stack dtype conflict are repaired.
- Active runtime evidence still shows `config_load_mode = manual_json_fallback` plus large missing/unexpected key counts, so checkpoint/runtime layout mismatch remains an open Line C object even though step-0 image-feature forward is no longer blocked.

# MINT v8.4 Line C — Runtime Compatibility Execution Spec

**Question**
> Can the current MINT/PaliGemma stack load, forward, and take a training step without changing the scientific object?

## Required repo-level work
1. Create `docs/contracts/runtime_compatibility_contract_v84.json`
2. Patch `external/MINT/lerobot_policy_mint/src/lerobot_policy_mint/modeling_mint.py`
   - inspect actual PaliGemma layout
   - normalize runtime handles through an adapter
   - support top-level and legacy `.model` image-feature access
3. Patch `scripts/mint/run_g8_mint_train.py`
   - stamp runtime provenance
   - gate full training on compatibility smoke success
4. Add `scripts/mint/run_g8_runtime_compat_smoke.py`
   - import, processor load, model load, adapter resolve, forward, backward
5. Patch `scripts/mint/run_tiny_retrain_confirmation.py`
   - distinguish `runtime_contract_broken` from teacher/readiness failures

## Allowed verdicts
1. `runtime_contract_status = repaired`
2. `wrapper/runtime layout remains incompatible`
3. `checkpoint layout remains incompatible after wrapper repair`
