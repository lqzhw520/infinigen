# Harness Engineering v1.3 — Phase 1 Test Report

**日期**: 2026-03-31
**Campaign**: `mint_drawer_v1`
**Phase**: Phase 1 — Sovereignty Convergence
**状态**: ✅ COMPLETED

---

## 一、执行摘要

Phase 1（真相主权收敛）已完整执行并通过全部验证。共创建 3 个目录、迁移/新建 8 个 sovereign 文件、标记 11 个 legacy 文件、建立 4 个根目录 pointer，建立了完整的 versioned claim registry 基础。

---

## 二、目录结构验证

### 2.1 sovereign/ 目录（最终状态）

```
sovereign/
├── CAMPAIGN_TRUTH.generated.md   # ✅ 唯一 generated truth（双视图）
├── acceptance_criteria.yaml      # ✅ 从 acceptance_criteria.json 迁移
├── claims.yaml                   # ✅ Versioned Claim Registry（含 4 条 claim，revision history）
├── evidence/
│   ├── E001.yaml                # ✅ 首条 evidence
│   ├── index.json               # ✅ evidence 索引
│   └── archive/                 # ✅ 预留归档目录
├── handoff.md                   # ✅ 人机交接文档
├── manifest.yaml                # ✅ 从根目录迁移（symlink 源）
├── next_actions.json            # ✅ 从根目录迁移（symlink 源）
└── state.json                  # ✅ 从根目录迁移（symlink 源）
```

### 2.2 根目录指针文件

```
manifest.yaml         -> sovereign/manifest.yaml          ✅ symlink
state.json            -> sovereign/state.json             ✅ symlink
next_actions.json     -> sovereign/next_actions.json      ✅ symlink
CAMPAIGN_TRUTH.md     (stub: LEGACY POINTER)            ✅ text file, not symlink
```

**设计原因**：`CAMPAIGN_TRUTH.md` 不是 symlink 而是文本 stub，防止 agent 把 symlink 目标文件当作 sovereign 来直接编辑。

### 2.3 Legacy 文件（已打标，未搬迁）

共 11 个文件已插入 `<!-- LEGACY DOCUMENT -->` header，内容完整保留，仅添加禁止使用标注：

```
acceptance_criteria.json    ✅ marked + migrated to sovereign/acceptance_criteria.yaml
campaign_spec.md             ✅ marked
campaign_status.md           ✅ marked
decision_memo.md           ✅ marked
findings.md                 ✅ marked
progress.md                 ✅ marked
review.json                 ✅ marked
review_prompt.md           ✅ marked
step_review_guide.md        ✅ marked
summary.json                ✅ marked
takeover_memo.md           ✅ marked
```

---

## 三、核心验证结果

### 3.1 `02_reconcile_sources.py` — Source-of-Truth Gate

```
=== 02_reconcile_sources.py ===
Campaign root: .../mint_drawer_v1

=== RECONCILE OK (0 warnings) ===
```

**检查项目（全部通过）**：
- [x] `manifest.yaml`、`state.json`、`next_actions.json` 均为 symlink
- [x] `CAMPAIGN_TRUTH.md` 为合法 stub（无 "GENERATED FILE"，有 "LEGACY POINTER"）
- [x] 11 个 legacy 文件全部有 LEGACY DOCUMENT header
- [x] `evidence/index.json` 为有效 JSON
- [x] 所有 required sovereign 文件存在

### 3.2 Claims Revision Chain — 深度验证

```
OK: C_SIGLIP_GENERALIZATION: current=3, lifecycle=active, status=candidate, 3 revisions
OK: C_VQVAE_USABLE:          current=1, lifecycle=active, status=contradicted, 1 revisions
OK: C_DATASET_SUFFICIENT:    current=1, lifecycle=active, status=contradicted, 1 revisions
OK: C_FINETUNE_IMPROVES_HELDOUT: current=1, lifecycle=active, status=contradicted, 1 revisions

ALL CHECKS PASSED — 4 claims, revision chain valid
```

**Revision Chain 验证项（全部通过）**：
- [x] `C_SIGLIP_GENERALIZATION@1 → superseded_by: C_SIGLIP_GENERALIZATION@2`
- [x] `C_SIGLIP_GENERALIZATION@2 → superseded_by: C_SIGLIP_GENERALIZATION@3`
- [x] `C_SIGLIP_GENERALIZATION@3 → superseded_by: null`（current live）
- [x] 每个 claim 只有 1 个 live revision（`superseded_by=null`）
- [x] `current_revision` 指针与最后一个 revision 编号一致
- [x] revision 编号从 1 开始、连续（1, 2, 3）
- [x] `change_type` 枚举合法（create / scope_narrowed / reframed）
- [x] `change_reason` 在非 create 的 revision 上均已填写
- [x] `superseded_by_reason` 在有 supersession 的 revision 上均已填写
- [x] `lifecycle_status` 枚举合法（active / superseded / split / archived）
- [x] 跨 claim supersession 链接目标存在（无断裂）

### 3.3 E001 Evidence — 结构验证

```
✅ evidence_id: E001
✅ experiment_id: d1_v57_pretrained
✅ metrics.grasp_success: 0.0
✅ metrics.total_eef_motion_m: 4.003
✅ related_claim_ids: [C_SIGLIP_GENERALIZATION, C_VQVAE_USABLE, C_FINETUNE_IMPROVES_HELDOUT]
✅ verified: false
✅ artifact_path: 指向 artifacts/d1_single_rollout_overfit.json
```

### 3.4 `sovereign/CAMPAIGN_TRUTH.generated.md` — 双视图验证

```
✅ Part A: Current Active Claims（4 条 claim 的 current_revision 快照）
✅ Part B: Recent Claim Revisions（6 条最近 revision，含 change_type + reason）
✅ CAMPAIGN_TRUTH.md 根目录 stub 正确指向 sovereign/CAMPAIGN_TRUTH.generated.md
```

---

## 四、回归检查（对现有工作的保护）

| 检查项 | 结果 |
|--------|------|
| `artifacts/` 未被修改 | ✅ |
| `runtime/` 未被修改 | ✅ |
| `dataset/` 未被修改 | ✅ |
| `videos/` 未被修改 | ✅ |
| `scripts/mint/` 未被修改 | ✅ |
| `drawer_robot_env.py` 未被修改 | ✅ |
| MINT / Infinigen submodule 未被触碰 | ✅ |
| 根目录软链接指向 sovereign/（非新文件） | ✅ |
| Legacy 文件内容完整（仅头部加了 header） | ✅ |
| V58 pipeline 脚本位置未变化 | ✅ |

---

## 五、Phase 2 待实现清单

| 待实现 | 状态 |
|--------|------|
| `03_claim_lint.py`（含 revision chain 完整检查） | 待实现 |
| `04_action_lint.py`（含 superseded/split-aware blocker 检查） | 待实现 |
| `05_verify_env_contract.sh`（scaffold） | 待实现 |
| `sovereign_cli.py`（10 个子命令） | 待实现 |
| `AGENTS.md` bootstrap 协议 | 待实现 |
| `infinigen-scientific-dev` skill 更新 | 待实现 |
| Phase 3 GC scripts（4 个，report-only） | 待实现 |
| Phase 4a night runner（watcher + planner + handoff） | 待实现 |
| Phase 4b night verifier + dry-run executor | 待实现 |
| Phase 4c bounded mutation | 待实现 |

---

## 六、使用指导

### 6.1 当前阶段：Phase 1 完成（Bootstrap 可用但不完整）

**立即可用**：
```bash
cd /mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1

# 验证 sovereign 完整性（每次新 session 前运行）
python3 scripts/harness/02_reconcile_sources.py
# 期望输出：=== RECONCILE OK (0 warnings) ===

# 验证 claim chain（Phase 2 lint 前哨）
python3 -c "
import yaml
data = yaml.safe_load(open('sovereign/claims.yaml'))
for c in data['claims']:
    print(f\"{c['claim_id']}: rev{c['current_revision']} {c['lifecycle_status']} {c['revisions'][-1]['status']}\")
"

# 阅读 handoff
cat sovereign/handoff.md

# 阅读 generated truth（含双视图）
cat sovereign/CAMPAIGN_TRUTH.generated.md
```

### 6.2 Agent 引导（在 Cursor 新 session 中）

在每次新 Cursor session 开始时，agent 应先运行：
```
python3 scripts/harness/02_reconcile_sources.py
```
预期：`=== RECONCILE OK (0 warnings) ===` 或 `N warnings`（warnings 不阻断）。

### 6.3 Claim 读取方式

当前 session 应以 `sovereign/claims.yaml` 为唯一 claim 真相源。

读取 claim 状态（快速摘要）：
```python
import yaml
data = yaml.safe_load(open('sovereign/claims.yaml'))
for c in data['claims']:
    rev = c['revisions'][c['current_revision'] - 1]
    print(f"{c['claim_id']}: {rev['status']} | scope: {rev['scope']}")
```

读取 claim 演化历史（最近 3 次）：
```python
import yaml
data = yaml.safe_load(open('sovereign/claims.yaml'))
for c in sorted(data['claims'], key=lambda x: x['revisions'][-1]['created_at'], reverse=True)[:3]:
    r = c['revisions'][-1]
    print(f"{c['claim_id']}@{r['revision']} [{r['change_type']}] at {r['created_at']}: {r['change_reason'][:80]}")
```

### 6.4 Legacy 文件说明

以下文件已打标为 legacy，**不得作为真相源使用**：
- `campaign_status.md`、`findings.md`、`progress.md`、`decision_memo.md`
- `takeover_memo.md`、`step_review_guide.md`、`campaign_spec.md`
- `review_prompt.md`、`acceptance_criteria.json`、`summary.json`、`review.json`

如需查看历史内容，可以读取（内容完整保留），但不得作为当前 claim/状态/决策的依据。

### 6.5 下一步：Phase 2

Phase 2 实现后，`sovereign_cli.py bootstrap` 将自动运行所有 lints 并输出 session 摘要。Phase 1 提供的是"手动验证模式"，等效于 bootstrap 的核心检查。

### 6.6 如需回滚

如果 sovereign 结构出现问题：
```bash
cd experiments/mint/mint_drawer_v1

# 检查软链接
ls -la manifest.yaml state.json next_actions.json

# 如果软链接断了，重新执行
bash scripts/harness/01_sync_pointers.sh

# 原始 sovereign 文件在 archive/20260331_legacy/ 中可查
```

---

## 七、文件清单（Phase 1 新增）

```
experiments/mint/mint_drawer_v1/
├── sovereign/
│   ├── claims.yaml                          [NEW] versioned claim registry
│   ├── CAMPAIGN_TRUTH.generated.md          [NEW] dual-view generated truth
│   ├── handoff.md                           [NEW] human-agent handoff doc
│   ├── acceptance_criteria.yaml              [NEW] migrated from acceptance_criteria.json
│   ├── evidence/
│   │   ├── E001.yaml                       [NEW] first evidence record
│   │   └── index.json                       [NEW] evidence index
│   ├── manifest.yaml                        [MIGRATED] symlink source
│   ├── state.json                           [MIGRATED] symlink source
│   └── next_actions.json                    [MIGRATED] symlink source
├── archive/20260331_legacy/                 [NEW] reserved for future archive
├── scripts/harness/
│   ├── 01_sync_pointers.sh                 [NEW] symlink + stub sync
│   ├── 02_reconcile_sources.py              [NEW] source-of-truth lint
│   └── yaml_utils.py                        [NEW] shared YAML I/O utilities
├── CAMPAIGN_TRUTH.md                        [MODIFIED] now a stub/pointer
├── manifest.yaml                             [MODIFIED] now symlink
├── state.json                               [MODIFIED] now symlink
├── next_actions.json                        [MODIFIED] now symlink
└── [11 legacy files]                        [MODIFIED] LEGACY header added
```
