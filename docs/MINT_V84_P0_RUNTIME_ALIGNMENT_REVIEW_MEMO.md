# MINT v84 P0 Runtime Alignment Review Memo

## Authority Snapshot
- Repo: `/mnt/afs2/zhuhaowu/infinigen`
- Branch: `feature/mint-env-reformulation-v1-visual-fidelity`
- Current authoritative `HEAD`: `b9c0ac7a14891959f7b8594ec9b03b2136ff3f6d`
- Vendor `HEAD`: `4eab5795345721001c412ff1ca2c886a11eab606`
- Active run: `canary_20260421T114811Z_b9c0ac7a_cecd75bf`
- Execution scope: `p0_canary_train_probe`
- Diagnostic-only: `true`
- Claim-bearing: `false`
- Primary spec used for the latest line: `/mnt/afs2/zhuhaowu/infinigen/docs/MINT_V84_P0_INFRASTRUCTURE_REPAIR_SPEC.md`

## What Remains Authoritative
The current P0 line completed in its own declared scope.

- `P0a`: `PASS`
- `P0b`: `PASS`
- `Re-C1`: `PASS`
- `Re-C2`: `PASS`
- `Re-C3`: `PASS`
- `Canary`: `PASS`
- Canary tier outcome: `TIER3_NO_SIGNAL_OR_P0C_REGRESSION`

The final canary deltas remain:
- `attach_bridge_gain = 0.0`
- `ever_attach_eligible_fraction_gain = 0.0`
- `ever_attached_rate_gain = 0.0`
- `stable_attach_rate_gain = 0.0`
- `phase_locked_rate_gain = 0.0`
- `close_cmd_rate_delta = -0.6875000029802323`
- `distance_pass_rate_gain = 0.0`
- `approach_gate_gain = +0.10416667163372043`

So the latest line still authoritatively says: inside the current benchmark, the canary did not produce attach signal and the close command regressed.

## What Changed In Interpretation
A later runtime audit showed that the current MuJoCo benchmark is not the same task object that the project has been implicitly comparing against the LIBERO drawer baseline.

The current runtime is:
- `MuJoCo` backend
- `drawer-only` scene model loaded from procedural drawer URDFs
- `proxy EEF / gripper state` maintained inside the environment
- task telemetry computed from proxy EEF state relative to the drawer handle geometry

This means the current line is **not** a full robot-in-scene LIBERO-style drawer task.

## Evidence For The Runtime Mismatch
From `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/drawer_robot_env_mujoco.py`:
- `_load_assets()` loads `drawer.urdf`, `metadata.json`, and `semantic_mapping.json` from `sim_exports/urdf/drawer/<seed>`.
- `mujoco.MjModel.from_xml_string(urdf_text, assets)` builds the scene directly from that drawer URDF payload.
- `reset()` directly assigns `self.eef_pos`, `self.eef_quat`, and `self.gripper_joint`.
- `step()` directly updates `self.eef_pos` and `self.gripper_joint` from the action before computing interaction gates.

A direct runtime check on the current authoritative environment produced:
- `env.model.ngeom = 41`
- sample geom names were drawer base / drawer door / drawer handle names only
- `has_panda_geom = false`

So the benchmark currently used for P0 canary is best described as:

```text
drawer-only MuJoCo scene
+ proxy EEF / gripper state machine
+ reach/approach/orientation/attach telemetry computed from that proxy state
```

## Why The Line Got Lost
The project did a lot of real alignment work, but most of it was alignment inside the proxy-runtime benchmark rather than alignment of the benchmark object itself to the intended LIBERO-style drawer task.

The iterations focused on:
- teacher dispatch
- feature computation parity
- state/action interface wiring
- one-step fit
- canary telemetry

But the line never froze a stronger prerequisite:
- scene-level parity with a full robot-in-scene drawer task
- asset / task-object parity with the intended baseline object
- rollout-video parity that a human reviewer could compare against LIBERO

That is why the line kept producing numerically meaningful but visually and semantically confusing results.

## What Still Matters From v11 / v12 / v13 / P0
These iterations were not useless.

They still established real facts about the current proxy-runtime benchmark:
- teacher family dispatch needed repair and was repaired
- feature computation path needed unification and was repaired
- live support expansion / parity / signal-band checks were made much stricter
- the current canary genuinely shows no attach gain and a strong close-command regression in this benchmark

What they do **not** currently justify is a stronger claim like:
- `MINT fails the LIBERO-aligned Infinigen drawer task`
- `P2 paradigm review is already scientifically compelled for the stronger full-robot task object`

## Current Claim Scope
The strongest defensible claim right now is:

> The latest P0 canary shows a Tier 3 outcome in the current `drawer-only MuJoCo + proxy EEF/gripper` benchmark. This is real for that benchmark, but it is not yet a clean empirical verdict about a LIBERO-aligned full-robot drawer task.

## Supporting But Unexecuted Follow-on Spec
A follow-on spec also exists at `docs/MINT_V84_P1_LEARNING_DYNAMICS_REPAIR_SPEC.md`. It should currently be read as a supporting diagnosis hypothesis, not as an executed line. It was not run after the stronger runtime-alignment mismatch was discovered.

The same applies to the remaining v13 helper-code cleanups that are about canonical state reconstruction and teacher-family dispatch reuse. They are being preserved on the branch as historical support code, not as evidence that a new v13/P1 execution line was completed.

## Questions For Science Agent
1. Should the current proxy-runtime benchmark be accepted as a legitimate claim-bearing task object, or should it be explicitly downgraded to an internal diagnostic environment?
2. If it is insufficient, what is the minimum runtime-parity repair spec needed before any future training claim can be interpreted against the intended drawer task?
3. Which existing findings survive unchanged after that runtime-parity correction, and which ones must be rerun?
4. What is the right empirical-claim ladder from here so that future results cannot drift from task-object mismatch into scientific overclaim again?
5. If a new runtime-parity gate is required, should it be treated as a pre-P0 blocker or as a new `P-1 runtime alignment` line?

## Recommended Immediate Discipline
- Do not continue training from the current line.
- Do not present the current report videos as full robot interaction evidence.
- Do not elevate `Tier 3` into a paradigm-level claim for the stronger LIBERO-style task object.
- Use the current line as a diagnostic benchmark closeout plus a runtime-alignment warning.
