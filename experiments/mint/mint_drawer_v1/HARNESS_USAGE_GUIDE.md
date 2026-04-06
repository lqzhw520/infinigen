# Harness v2 — mint_drawer_v1
**Campaign**: `mint_drawer_v1`
**Status**: `v59_LEARNABILITY_AUDIT_IN_PROGRESS`
**Phase**: Learnability Audit
**Updated**: 2026-04-06T14:35+08:00
**STALE WARNING**: This file must be updated at end of each session.

---

## 当前 Verdict（每次 session 开始时确认）

```
V59_RCA123_ALL_ELIMINATED — P0b_DRAWER_COLOR_ELIMINATED — SIMULATOR_PHYSICS_NOW_PRIMARY
```

**完整 Truth Sources**（只读这些）：
```
sovereign/SESSION_BOOTSTRAP.20260406.md ← 每次 session 第一个读的文件
sovereign/evidence/E022.yaml  — Env gate: white=0/6
sovereign/evidence/E023.yaml — RCA1: teacher ELIMINATED (7/7 successful)
sovereign/evidence/E024.yaml — RCA2: action normalization ELIMINATED
sovereign/evidence/E025.yaml — RCA3: state representation ELIMINATED
sovereign/evidence/E026.yaml — P0b: drawer color ELIMINATED (matched A/B: 0/6)
sovereign/next_actions.json — 当前队列
sovereign/state.json — phase + verdict（Tier 1，不含 narrative）
sovereign/run_ledger.yaml — session 历史
```

**过时的文件（不要作为当前依据）**：
```
sovereign/handoff.md        ← 停在了 2026-04-04
HARNESS_USAGE_GUIDE.md    ← 停在了 2026-04-05
sovereign/HARNESS_HYGIENE.md ← 停在了 2026-04-05
sovereign/CAMPAIGN_TRUTH.*  ← 旧的 generated view
state.json (legacy fields) ← V58 narrative 已废弃
artifacts/p0b_colored_rollouts/seed_007_* ← exploratory superseded
```

---

## 新 Session 启动（每个 Cursor session 开始时）

```bash
cd /mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1

# 1. Read bootstrap FIRST
cat sovereign/SESSION_BOOTSTRAP.20260406.md

# 2. Reconcile sovereign
python3 scripts/harness/02_reconcile_sources.py
# 期望: RECONCILE OK

# 3. Run lints
python3 scripts/harness/03_claim_lint.py
python3 scripts/harness/04_action_lint.py
```

---

## 当前 Learnability Audit 进度

| Gate | 名称 | 状态 | 证据 |
|------|------|------|------|
| Gate A | Episode Admissibility | ⏳ **PENDING** | — |
| Gate B | Teacher Replayability | ⏳ **PENDING** (critical gap) | — |
| Gate C | Visual Sufficiency | ⏳ PENDING | — |
| Gate D | Model Load Fidelity | ⚠️ 部分完成 | F1 rated, not canonical |
| Gate E | Task Learnability | ⏳ PENDING | — |
| RCA1 | Teacher Quality | ✅ ELIMINATED | E023: 7/7 successful |
| RCA2 | Action Normalization | ✅ ELIMINATED | E024 |
| RCA3 | State Representation | ✅ ELIMINATED | E025 |
| P0b | Drawer Color (sole) | ✅ ELIMINATED | E026: 0/6 matched A/B |

---

## 立即下一步

**Gate A: Episode Admissibility Audit** — 跑全量 240 episodes 的 episode-level 诊断，分类：
- `task-teaching`: strict_success=True, ever_attached=True, meaningful attach persistence
- `motion-only`: 有动作但无 attach
- `weak/noisy`: 低 action variance、状态异常、physics illegal

**Gate B: Teacher Replayability**（在 Gate A 后执行）— 验证 teacher action 在 DrawerRobotEnv 中是否可重放成功。**这是当前最大缺口**。如果不过，后面任何 MINT 调优都是在拟合一个当前 env 不可执行的目标。

---

## Run Ledger

```
sovereign/run_ledger.yaml ← append-only，每次 session 追加
```

**最近 session**：`run_001_p0b_matched_ab`（2026-04-06, 1h, drawer color eliminated）

---

## Phase 进度

| Phase | 内容 | 状态 |
|-------|------|------|
| V58 Train/Eval | V58 finetune + eval | ✅ 完结（0%） |
| V59 Data Pack | 240ep, 19701f, dataset_loads=true | ✅ 完结 |
| RCA1/2/3 | Teacher/Action/State elimination | ✅ 完结 |
| P0b Drawer Color | Matched A/B | ✅ 完结 |
| **Learnability Audit** | Gate A/B/C/D/E | ⏳ **IN PROGRESS** |
| Simulator Physics | 待定 | ⏳ 待定 |
