# MINT v8.4 Unified Execution Spec (v2 — Codex + GPT-5.4 Dual Review)

**Authority**
- Repo: `/mnt/afs2/zhuhaowu/infinigen`
- Branch: `feature/mint-env-reformulation-v1-visual-fidelity`
- Current sovereign head: `72711fbfc9ddd5781aa130035bdc32c2e0ea5489`
- Authoritative sovereign for code: `fea75fb436b40bd5f854d7a74063b8ecff663cd4`
- `external/MINT` head: `137b42d627c308d4fc1cb6b1f84e92a5a7892b74`

**Canonical companion state snapshot**
- `MINT_V84_SYSTEM_AUDIT_2026-04-16.md`
- `docs/contracts/truth_contract_v84.json`
- `docs/contracts/runtime_compatibility_contract_v84.json`
- `docs/contracts/acceptance_contract_v84.json`
- `docs/contracts/teacher_frontier_reference_v84.json`
- `docs/contracts/t3a_frozen_reference_v84.json`

**Supersedes and replaces as execution entrypoints**
- `MINT_V84_BOUNDED_MAINLINE_EXECUTION_SPEC.md`
- `MINT_V84_FULL_FIX_SPEC.md`
- `MINT_V84_FULL_FIX_EXECUTION_PLAN.md`
- `MINT_V84_LINEA_T3B_REBUILD_EXECUTION_SPEC.md`
- `MINT_V84_LINEB_TRUTH_CONTRACT_ALIGNMENT_EXECUTION_SPEC.md`
- `MINT_V84_LINEC_RUNTIME_COMPATIBILITY_EXECUTION_SPEC.md`

**Authors**: Codex (Claude 4.6 Opus Max Thinking) + GPT-5.4 Pro Expert dual review
**Date**: 2026-04-16

---

## 0. 双重 Review 的分工与合并依据

本 spec 合并了两套独立 review 的发现，按来源分工：

| 来源 | 主要贡献 |
|------|----------|
| Codex（我） | Line A 代码级 bug 定位（code path analysis） |
| GPT-5.4 | Line B/C governance bug（registry/enforcement analysis） |
| 共同确认 | Line A scientific object convergence；Regime-vs-acceptance pressure |

双方的定位均已与实际代码/artifact 对照；但文中列出的修复项需明确区分“已被定位”和“已进入当前 sovereign 代码”，不得混写。

---

## 1. 当前全局状态

### 1.1 三条线的真实状态（修正版）

#### Line B — **materially aligned，registry schema closure 仍是待落地修复**

行为层面已对齐（`dataset_valid=true`，`errors=[]`），但当前 sovereign 代码仍在使用 `teacher_predicate` + fallback；registry schema closure 已被定位（见 §3.2），但尚未进入 authoritative code。

#### Line C — **first-forward barrier crossed，smoke gate hardening 仍是待落地修复**

本轮 smoke 到达 `image_features_resolved`，但当前 sovereign 代码仍把 `image_features_failed` 计为 passed，且 smoke 还未推进到 contract 所要求的 forward/backward stages（见 §4.2）。

#### Line A — **scientific object 已收敛，verdict 逻辑和 controller 实现均未就位**

最新 audit 把科学问题精化为 **reset-to-frontier continuation under partially-open geometry**。但：
- `gap_to_frontier` 记录在 `evidence` 里，未进入 `clauses` verdict（GPT 发现）
- RCA class 标签仍为旧的 `"interaction_frame_not_superior"` 等，未更新（GPT 发现）
- controller 实现存在 `hybrid_frame_*` freeze bug，导致 second burst 无法 reattach（我定位）

### 1.2 上游双压力（双方共同确认）

```
Line A pressure:     reset gap (R_s below U_s)      → 用 T3B-r4 修
Regime pressure:     frontier below acceptance        → 最小 paradigm shift
                       U_2=0.690 / U_4=0.568
                   acceptance:  near=0.85 / strict=0.90
```

### 1.3 决策树（双方共同确认）

```
T3B-r4 执行后
    │
    ├─ reset gap 仍未收敛到 frontier
    │       → "reset abstraction still primary bottleneck"
    │
    ├─ reset gap 收敛，但 frontier 仍低于 acceptance
    │       → 停止 controller tweak，转 "最小 grasp-family diversification frontier assay"
    │
    └─ reset gap 收敛，frontier 达到或接近 acceptance
            → 继续 Line B/C → integrated rerun
```

---

## 2. Line A — T3B-r4 Continuation Rebuild

### 2.1 Scientific Object（已收敛，双方确认）

> 当前 T3B 的问题不是"单步 opening authority 不够"（第一步 burst 的 step authority 已够强）。  
> 当前问题是：**第一段 partially-open burst 结束后，无法 reattach/relock 并开启第二段 opening burst。**  
> 科学对象：`reset-to-frontier continuation under partially-open geometry`

### 2.2 Controller Bug — P1（Codex 定位；Fix A/B/C 为 T3B-r4 必做修复）

#### 根因链（代码级，已逐行验证）

```
reseat 后 EEF 物理位置大幅改变
        ↓
hybrid_frame_n/b 在第一次 grasp 时 freeze，后续一直强制复用
        ↓
interaction_n = hybrid_frame_n 方向已经和当前几何不匹配
        ↓
target_pos = handle + axis*s_t + interaction_n*s_n 计算错误
        ↓
_script_action 的 pos_err 偏 → delta 方向偏
_script_action 的 target_quat 偏（通过 target_pos 传导）
        ↓
EEF quat orientation 偏差导致 orientation_alignment_cos < 0.60
        ↓
phase_locked = False → last_phase_locked = False
        ↓
hybrid_lock_score 上限只剩 0.35，无法积累到阈值
        ↓
interaction_lock 的 hybrid_lock_streak 永远达不到 3
        ↓
永远卡在 interaction_lock，无法进入 hybrid_open_ramp
        ↓
second burst 永不开始 → plateau at partially-open state
```

#### 关键代码位置

```3237:3240:scripts/mint/drawer_robot_env_mujoco.py
# 有 bug：每次 grasp_seat/interaction_lock 都重新采样 frame（抖动）
# 但 reseat 后 EEF 位置已变，frame 方向已经偏了
elif phase in {"grasp_seat", "interaction_lock"} and env._attached:
    hybrid_frame_n = interaction_n.copy()
    hybrid_frame_b = interaction_b.copy()

3641:3653:scripts/mint/drawer_robot_env_mujoco.py
# 有 bug：reseat_once 阶段强制用 freeze 的旧 frame
elif (
    phase
    in {"hybrid_open_ramp", "hybrid_open_hold",
        "hybrid_open_final", "reseat_once"}  # ← reseat_once 不该在这里
    and hybrid_frame_n is not None
    and hybrid_frame_b is not None
):
    interaction_n = hybrid_frame_n.copy()   # ← 用旧 frame，方向已偏
    interaction_b = hybrid_frame_b.copy()
```

#### 修复方案（三处修改）

**Fix A — reseat 后重置 frame**

在 `reseat_once` transition 完成后，将 `hybrid_frame_n/b` 设为 `None`，强制下次 `interaction_lock` 进入时重新捕获：

```3453:3464:scripts/mint/drawer_robot_env_mujoco.py
elif phase == "reseat_once":
    micro_retract_phase_steps += 1
    if (
        np.linalg.norm(obs.eef_pos - micro_retract_target) < 0.015
        or micro_retract_phase_steps >= 2
    ):
        hybrid_reseat_used = True
        hybrid_lock_streak = 0
        hybrid_positive_streak = 0
        hybrid_no_progress_streak = 0
        hybrid_high_slip_streak = 0
        # --- 新增一行：强制 reseat 后重新捕获 frame ---
        hybrid_frame_n = None
        hybrid_frame_b = None
        phase = "interaction_lock" if env._attached else "contact"
```

**Fix B — 移除 reseat_once from freeze list**

从强制 freeze 的 phase 列表中移除 `reseat_once`：

```3641:3653:scripts/mint/drawer_robot_env_mujoco.py
# 修复后：只在 burst 期间保持 frame 稳定，reseat 期间允许重新采样
elif (
    phase
    in {
        "hybrid_open_ramp",
        "hybrid_open_hold",
        "hybrid_open_final",
    }
    and hybrid_frame_n is not None
):
    interaction_n = hybrid_frame_n.copy()
    interaction_b = hybrid_frame_b.copy()
# reseat_once 不再强制 freeze → 让 grasp_seat → interaction_lock 重新捕获
```

**Fix C — 在 interaction_lock 首次进入时捕获 frame**

将 frame 捕获从 `grasp_seat/interaction_lock` 每次执行改为只在 `interaction_lock` 首次进入时一次：

```3237:3240:scripts/mint/drawer_robot_env_mujoco.py
# 修复后：只在 interaction_lock 首次进入时捕获（gated by hybrid_frame_n is None）
if phase == "interaction_lock" and env._attached and hybrid_frame_n is None:
    hybrid_frame_n = interaction_n.copy()
    hybrid_frame_b = interaction_b.copy()
```

**核心原则**：每个 opening burst 有自己的 interaction frame，burst 之间不共享。

#### 澄清：P2 不是真实问题（Codex 验证）

GPT 认为 reseat 后跳过了 `grasp_seat` 阶段。经验证代码，reseat 后路径为：

```
reseat_once → contact → _attached=True → grasp_seat 触发（行3263-3267）
```

`grasp_seat` **不会被跳过**，GPT 的 P2 诊断是误判。但 `grasp_seat` 被触发后，由于 Fix A/B/C 的 frame 问题，第二次 grasp 的 orientation 仍然会偏，这是 P1 的下游效应，不是独立的 P2。

### 2.3 Verdict Contract — 升级到 Frontier-Closure Language（GPT 定位；当前 sovereign 尚未落地）

#### 当前问题

`gap_to_frontier` 和 `gap_closed_fraction` 已记录在 `evidence` 里（行 1010-1011），但**不在 `clauses` 里作为 verdict 条件**。当前 verdict clauses 仍然是：

```963:975:scripts/mint/run_v84_teacher_abstraction_full.py
clauses = {
    "seed2_near_strict": bool(seed_results["2"]["has_accepted_teacher"]),
    "seed4_near_strict": bool(seed_results["4"]["has_accepted_teacher"]),
    "at_least_one_seed_strict": bool(...),
    "matched_superiority_over_t3a": bool(...),  # ← 旧语言
}
```

这意味着 verdict governance 和 scientific object 是脱节的。

#### 修复方案

T3B 的 clauses 改为 frontier-closure 语言（新增，不删除旧条款作为并行参考）：

```python
# T3B-r4 verdict clauses
clauses = {
    # 旧条款（保留作为并行参考）
    "seed2_near_strict": bool(seed_results["2"]["has_accepted_teacher"]),
    "seed4_near_strict": bool(seed_results["4"]["has_accepted_teacher"]),
    "at_least_one_seed_strict": bool(...),
    "matched_superiority_over_t3a": bool(...),
    # --- 新增：frontier-closure 语言（主 verdict） ---
    "gap_closed_seed2_ge_0p7": float(frontier_gap_closed_fraction.get("2") or -1) >= 0.7,
    "gap_closed_seed4_ge_0p7": float(frontier_gap_closed_fraction.get("4") or -1) >= 0.7,
    "approaches_frontier_seed2": float(frontier_gap.get("2") or 999) <= 0.05,
    "approaches_frontier_seed4": float(frontier_gap.get("4") or 999) <= 0.05,
}
```

#### RCA Classes — 更新为 Continuation Language（GPT 定位，已验证）

当前 RCA class 标签（行 1021-1025）仍为旧的 `"interaction_frame_not_superior"`，需更新：

```1018:1026:scripts/mint/run_v84_teacher_abstraction_full.py
rca_classes=_rca_from_mapping(
    clauses,
    {
        "seed2_near_strict": "continuation_failure_seed2_after_first_burst",
        "seed4_near_strict": "continuation_failure_seed4_after_first_burst",
        "at_least_one_seed_strict": "strict_threshold_not_reached",
        "matched_superiority_over_t3a": "continuation_not_superior_to_world_frame",
        # --- 新增 ---
        "gap_closed_seed2_ge_0p7": "reset_gap_seed2_not_closed",
        "gap_closed_seed4_ge_0p7": "reset_gap_seed4_not_closed",
        "approaches_frontier_seed2": "frontier_gap_seed2_large",
        "approaches_frontier_seed4": "frontier_gap_seed4_large",
    },
)
```

### 2.4 Required Telemetry

```python
# 新增 trace 字段
"hybrid_frame_n_trace"      # interaction_n 的实际方向（用于 debug frame freeze bug）
"hybrid_reseat_triggered"   # reseat 是否被触发
"hybrid_reseat_phase_before" # reseat 前的 phase
"second_burst_started"       # 是否进入了第二次 hybrid_open_ramp
"second_burst_max_drawer_delta"  # 第二次 burst 的最大 step delta
"continuation_plateau_reason"    # plateau 原因（从 hybrid_open_hold 的 controller_plateau_reason 继承）
```

### 2.5 Required Run Order

```bash
python scripts/mint/run_v84_teacher_abstraction_full.py --phase p1a
python scripts/mint/run_v84_teacher_abstraction_full.py --phase p1b
python scripts/mint/run_v84_teacher_abstraction_full.py --phase t3a
python scripts/mint/run_v84_teacher_abstraction_full.py --phase t3b --resume-from t3a
```

### 2.6 Allowed Verdicts

| verdict | 含义 |
|---------|------|
| `RESET_ABSTRACTION_STILL_PRIMARY` | reset gap 未显著收敛 |
| `RESET_TO_FRONTIER_CLOSED_BUT_FRONTIER_BELOW_ACCEPTANCE` | reset gap 收敛但 regime 压力仍在 |
| `RESET_SUPPLY_RESTORED_AND_FRONTIER_REACHED` | reset gap 收敛且 frontier 接近 acceptance |

---

## 3. Line B — Truth-Contract Schema Closure（GPT 定位；当前 sovereign 尚未落地）

### 3.1 Root Cause

GPT 发现：`tiny_retrain_mainline.py` 访问 `truth_contract` 时使用的是 **`teacher_predicate`**（行 369、744），而 `truth_contract_v84.json` 定义的是 **`teacher_truth_predicate`**。字段名不匹配导致永远走到 `or "teacher_window_truth_v84"` fallback。

这造成的后果：行为上碰巧对齐（因为 fallback 值恰好等于 registry 值），但**代码完全没有被 registry 约束**——后续如果 registry 值改变，代码不会同步。

### 3.2 Required Fixes

**Fix 1 — `tiny_retrain_mainline.py` 行 369**

```python
# 当前（错误）
"teacher_truth_adjudication": str(
    truth_contract.get("teacher_predicate") or "teacher_window_truth_v84"
),
# 修复后
"teacher_truth_adjudication": str(
    truth_contract["teacher_truth_predicate"]  # 直接读，无 fallback
),
```

**Fix 2 — `tiny_retrain_mainline.py` 行 744**

```python
# 当前（错误）
_truth_contract_payload().get("teacher_predicate") or "teacher_window_truth_v84"
# 修复后
_truth_contract_payload()["teacher_truth_predicate"]
```

**Fix 3 — `truth_contract_v84.json` 增强 forbidden_fallbacks**

```json
"forbidden_fallbacks": [
    "teacher_predicate",
    "dataset_predicate",
    "probe_predicate",
    "claim_predicate",
    "proxy_handle_hull",
    "top_level_summary_only_truth",
    "legacy_truth_gap_sentinel"
]
```

**Fix 4 — `tiny_retrain_mainline.py` 添加 hard-fail on missing**

在所有 `truth_contract["xxx_predicate"]` 访问外包裹：

```python
def _require_contract_key(contract: dict, key: str) -> str:
    if key not in contract:
        raise KeyError(
            f"truth_contract missing required key '{key}'. "
            f"Contract path: {TRUTH_CONTRACT_PATH}. "
            f"Allowed keys: {list(contract.keys())}. "
            f"No fallback permitted per v84 governance."
        )
    return str(contract[key])
```

### 3.3 Line B 阶段扩展

当前 Line B 已 materally aligned（`dataset_valid=true`）。修复后升级为 **authoritatively aligned**：

| 阶段 | 状态 | 条件 |
|------|------|------|
| materally aligned | 当前已满足 | behavior 对齐 |
| authoritatively aligned | 本次修复后达到 | schema 完全闭合 |

---

## 4. Line C — Runtime Smoke Gate Hardening（GPT 定位；当前 sovereign 尚未落地）

### 4.1 Root Cause

`run_g8_runtime_compat_smoke.py` 存在两个独立 bug：

**Bug 1 — `passed` 条件过宽**

```97:100:scripts/mint/run_g8_runtime_compat_smoke.py
report["passed"] = report["stage"] in {
    "image_features_resolved",
    "image_features_failed",   # ← 即使 image features 失败也算 passed！
}
```

**Bug 2 — `strict=False` 掩盖 checkpoint mismatch**

```64:66:scripts/mint/run_g8_runtime_compat_smoke.py
policy = MINTPolicy.from_pretrained(
    MINT_CKPT, config=config, local_files_only=True, strict=False
)
```

### 4.2 Required Fixes

**Fix 1 — 修正 `passed` 条件**

```python
# 修复后
report["passed"] = report["stage"] == "image_features_resolved"
# image_features_failed 不再算 passed
```

**Fix 2 — 记录 checkpoint mismatch 详情**

```python
# 修复后：在 policy load 后添加
if hasattr(policy, "model"):
    state_keys = set(policy.model.state_dict().keys())
    # 记录 missing/unexpected，不强制 fail（待 Line C 第二阶段处理）
    report["checkpoint_keys_recorded"] = True
```

**Fix 3 — 对齐 `runtime_compatibility_contract_v84.json` 和实际 smoke**

`runtime_compatibility_contract_v84.json` 定义的 smoke tests 包含 `forward_one_batch` 和 `backward_one_step`，但当前 smoke 只到 `image_features_resolved`。这是 spec 和实现之间的断层。

修复后的 smoke 应增加：

```python
# 在 image_features_resolved 后继续
try:
    # forward one batch
    img_features = img_features.to(policy.device)
    logits = policy.model(
        pixel_values=img_features,
        input_ids=torch.zeros((1, 32), dtype=torch.long, device=policy.device),
    )
    report["forward_resolved"] = True
except Exception as fwd_exc:
    report["forward_error"] = repr(fwd_exc)
    report["stage"] = "forward_failed"

# backward one step（可选，作为 smoke 的最后一步）
try:
    loss = logits.logits.sum()
    loss.backward()
    report["backward_resolved"] = True
except Exception as bwd_exc:
    report["backward_error"] = repr(bwd_exc)
    report["stage"] = "backward_failed"
```

### 4.3 Line C 分阶段

| 阶段 | 目标 | 当前状态 |
|------|------|----------|
| Stage 1 | first-forward barrier crossed | ✅ 已达到（`image_features_resolved`） |
| Stage 2 | checkpoint key mismatch 完整记录 | ❌ 需修复（`strict=False` 掩盖） |
| Stage 3 | `forward_one_batch` 可执行 | ❌ 待推进 |
| Stage 4 | `backward_one_step` 可执行 | ❌ 待推进 |

**注意**：本 spec 承认 Line C Stage 1 已跨过，但 Stage 2/3/4 仍未闭合。严禁将 Stage 1 误读为"runtime compatibility 已解决"。

---

## 5. Regime Pressure — 下一步决策（双方共同确认）

### 5.1 Frozen Reference

```
FROZEN_P1B_FRONTIER:
  seed2: 0.6899748044108651
  seed4: 0.5679305979991468

ACCEPTANCE:
  near:   0.85
  strict: 0.90
```

### 5.2 Decision Gate

```
T3B-r4 执行完毕
    │
    ├── gap_closed_fraction >= 0.7 on both seeds
    │   │
    │   ├── frontier >= acceptance threshold
    │   │       → PROCEED: Line B/C → integrated rerun
    │   │
    │   └── frontier < acceptance threshold
    │           → STOP controller tweaking
    │           → 转 §5.3：最小 paradigm shift
    │
    └── gap_closed_fraction < 0.7 on either seed
            → "reset abstraction still primary"
            → 诊断：是 second-burst failure（Fix A/B/C 没修到位）
              还是其他根因
```

### 5.3 最小 Paradigm Shift — Grasp-Family Diversification Frontier Assay

如果 T3B-r4 把 reset gap 补到 frontier，但 frontier 仍低于 acceptance，**不要再调 controller**。此时唯一合理的下一步是：

> 在保持 scripted contact-phase law 不变的前提下，引入 3–5 个离散的 grasp family / anchor family，测试 frontier assay 是否能在 hard seeds 上跨过 acceptance。

这不是"再调一个 controller"，而是**改变 regime 本身的接触模式集合**。Shared grasp 文献指出，接触任务的成功率高度依赖可行接触模式的集合。单一 fixed-grasp 可能本来就是 regime 上界偏低的原因。

---

## 6. Governance Rules（强制约束）

1. **不新增 branch**。所有修改在当前 `feature/mint-env-reformulation-v1-visual-fidelity` 执行。
2. **不新增 evidence family**。本 spec 覆盖所有活跃的 bounded lines。
3. **不放松 acceptance thresholds**。`near=0.85`，`strict=0.90` 冻结到本轮 bounded execution 结束。
4. **Line B registry 只从 schema 读**。禁止 `get(key) or fallback` 模式。
5. **Line C smoke `passed` 只在完全成功时为真**。`image_features_failed` 不再算 passed。
6. **T3B verdict 只接受 frontier-closure 语言作为主 verdict**。旧条款保留但仅作并行参考。
7. **Line A/B/C 各只执行一次 bounded run**，然后按 §5.2 决策树决策。
8. **不在 controller 层无限循环**。T3B-r4 之后如果 reset gap 仍存在，直接触发 regime-pressure 讨论。

---

## 7. Required Modifications Summary

| 文件 | 修改 | 对应 Line | 性质 |
|------|------|-----------|------|
| `drawer_robot_env_mujoco.py` 行 3237-3240 | frame 捕获改为 interaction_lock 首次进入时一次 | A | Bug fix |
| `drawer_robot_env_mujoco.py` 行 3453-3464 | reseat 后 `hybrid_frame_* = None` | A | Bug fix |
| `drawer_robot_env_mujoco.py` 行 3641-3653 | 移除 `reseat_once` from freeze list | A | Bug fix |
| `run_v84_teacher_abstraction_full.py` 行 963-975 | 新增 frontier-closure clauses | A | Verdict upgrade |
| `run_v84_teacher_abstraction_full.py` 行 1018-1026 | 更新 RCA class 标签为 continuation 语言 | A | Verdict upgrade |
| `tiny_retrain_mainline.py` 行 369 | `teacher_predicate` → `teacher_truth_predicate`，删 fallback | B | Schema closure |
| `tiny_retrain_mainline.py` 行 744 | 同上 | B | Schema closure |
| `truth_contract_v84.json` | 增强 `forbidden_fallbacks` | B | Schema closure |
| `run_g8_runtime_compat_smoke.py` 行 97-100 | `passed` 条件收紧 | C | Smoke gate |
| `run_g8_runtime_compat_smoke.py` 行 64-66 | checkpoint mismatch 完整记录 | C | Smoke gate |
| `runtime_compatibility_contract_v84.json` | 对齐 Stage 3/4 smoke 目标 | C | Spec/implementation alignment |

---

## 8. Canonical Execution Order

This document is the **only canonical execution spec** for current `v8.4`. The only other active document is `MINT_V84_SYSTEM_AUDIT_2026-04-16.md`, which is a state snapshot rather than an execution spec.

### Step 0 — Sovereign synchronization
1. Verify repo root, branch, top-level head, and `external/MINT` head.
2. Verify contract hashes for acceptance, truth, frontier, runtime, and frozen `T3A` reference.
3. Reject stale artifacts whose recorded head/contract hash does not match current sovereign.

### Step 1 — Line A first, because it is still the primary scientific blocker
1. Apply `T3B-r4` controller fixes only:
   - Fix A: clear `hybrid_frame_*` after `reseat_once`
   - Fix B: remove `reseat_once` from the freeze set
   - Fix C: capture burst frame only on first `interaction_lock` entry
2. Upgrade `T3B` verdict governance to frontier-closure language:
   - `gap_closed_seed2_ge_0p7`
   - `gap_closed_seed4_ge_0p7`
   - `approaches_frontier_seed2`
   - `approaches_frontier_seed4`
3. Keep `T3A` fidelity exact against frozen reference.
4. Run in this order only:
   - `p1a`
   - `p1b`
   - `t3a`
   - `t3b --resume-from t3a`
5. Interpret only three Line A verdicts:
   - `RESET_ABSTRACTION_STILL_PRIMARY`
   - `RESET_TO_FRONTIER_CLOSED_BUT_FRONTIER_BELOW_ACCEPTANCE`
   - `RESET_SUPPLY_RESTORED_AND_FRONTIER_REACHED`

### Step 2 — Line B schema closure, only after Line A rerun
1. Replace `teacher_predicate` reads with hard required `teacher_truth_predicate`.
2. Remove fallback reads from truth contract lookup.
3. Harden contract loading with required-key failures.
4. Rerun `prepare` and verify:
   - `dataset_valid = true`
   - `errors = []`
   - `teacher_truth_gate` comes from registry, not fallback

### Step 3 — Line C smoke hardening
1. Tighten smoke `passed` to `stage == image_features_resolved` only.
2. Record checkpoint mismatch explicitly instead of hiding it behind `strict=False`.
3. Extend smoke toward contract stages:
   - image features
   - forward one batch
   - backward one step
4. Only if wrapper/layout remains stable should checkpoint-remap work begin.

### Step 4 — Integrated rerun gate
Proceed to integrated rerun only if:
- Line A no longer reports `RESET_ABSTRACTION_STILL_PRIMARY`
- Line B is authoritatively aligned
- Line C reaches its intended smoke stage without false pass semantics

### Step 5 — Regime-pressure decision
If Line A closes reset-to-frontier but the frozen frontier remains below acceptance, stop controller tweaking. The next object is not another controller variant but a **minimal grasp-family diversification frontier assay**.

### Step 6 — Prohibitions
- No new controller family beyond `T3B-r4` in this round
- No acceptance relaxation
- No fallback-based truth reads
- No diagnostic artifact reuse across mismatched sovereign heads
- No parallel execution specs outside this document
