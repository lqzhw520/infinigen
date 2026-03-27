# Campaign Spec

MINT drawer AnyGrasp-conditioned robot-trajectory claim-push campaign.

- Train seeds: `1-10`
- Held-out seeds: `11-15`
- Active root-cause ladder:
  - `L1`: oracle scripted robot open
  - `L2`: AnyGrasp scripted robot open
  - `L3`: AnyGrasp-conditioned replay-faithful robot trajectories are learnable on train seeds
  - `L4`: fine-tuned MINT beats pretrained on held-out seeds
- Current overnight rule: do not stop at weaker claims before all C-layer repair branches and train-side probes are exhausted.
