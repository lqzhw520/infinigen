Current authoritative status:
- Wrapper adapter for top-level vs legacy PaliGemma image-feature access is implemented.
- `scripts/mint/run_g8_runtime_compat_smoke.py` now exists and runs.
- Current smoke evidence shows top-level layout is present, but config loading needs manual fallback and checkpoint/runtime mismatch is still active.

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
