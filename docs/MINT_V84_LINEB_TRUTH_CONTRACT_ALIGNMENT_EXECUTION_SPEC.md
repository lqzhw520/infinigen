# MINT v8.4 Line B — Truth-Contract Alignment Execution Spec

**Question**
> Do teacher acceptance, dataset inclusion, train-probe interpretation, and held-out claim use the same truth object family?

## Frozen object
- one contract family only: `truth_contract_v84`

## Required repo-level work
1. Create `docs/contracts/truth_contract_v84.json`
2. Patch `scripts/mint/tiny_retrain_mainline.py`
   - adjudicate teacher training truth from trace-rooted windows
   - emit training-truth metadata without collapsing to final snapshot truth
3. Patch `scripts/mint/dataset_builder.py`
   - validate against training truth and contract hash
   - separate rejection reasons
4. Patch `scripts/mint/run_tiny_retrain_confirmation.py`
   - report raw supply vs post-validation supply vs contract mismatch
5. Patch `scripts/mint/evaluate_mint_drawer_campaign_mujoco.py`
   - stamp probe/eval artifacts with the same truth contract hash

## Important execution constraint
- First adjudicate and index truthful windows.
- Only slice rollout sequences physically after index-based window validation is stable.

## Allowed verdicts
1. `truth_alignment_status = aligned`
2. `truth_alignment_status = misaligned`
