# Harness Engineering v1.3 — Phase 2 Test Report

**日期**: 2026-03-31 / 2026-04-01
**Campaign**: `mint_drawer_v1`
**Phase**: Phase 2 — Deterministic Gates + Controlled CLI
**状态**: ✅ COMPLETED

---

## 一、Phase 2 新增文件清单

```
scripts/harness/
├── 03_claim_lint.py                 ✅ NEW — Claim consistency + revision chain gate
├── 04_action_lint.py               ✅ NEW — Action queue gate (superseded-aware)
├── 05_verify_env_contract.sh        ✅ NEW — Env contract scaffold
├── sovereign_cli.py                ✅ NEW — 唯一受控写入 CLI（10 个子命令）
└── [phase1 files retained]
    ├── 01_sync_pointers.sh
    ├── 02_reconcile_sources.py
    └── yaml_utils.py
```

---

## 二、验证结果总表

| 检查项 | 预期 | 实际 | 结果 |
|--------|------|------|------|
| `02_reconcile_sources.py` | 0 ERROR | 0 ERROR, 0 WARNING | ✅ |
| `03_claim_lint.py` | 0 ERROR | 0 ERROR, 5 WARNING | ✅ |
| `04_action_lint.py` | 0 ERROR | 0 ERROR, 0 WARNING | ✅ |
| `05_verify_env_contract.sh` | scaffold OK | scaffold OK (deferred hard thresholds) | ✅ |
| `sovereign_cli.py bootstrap` | PASS | PASS — 3/3 lints OK | ✅ |
| `sovereign_cli.py reconcile` | PASS | PASS — 3/3 lints OK | ✅ |
| `sovereign_cli.py render-truth` | Valid markdown | Valid markdown, dual-view | ✅ |
| `sovereign_cli.py` 子命令 help | 可用 | 全部 10 个子命令 help 可用 | ✅ |
| `record-evidence` (missing file) | FATAL | FATAL + exit 1 | ✅ |
| `close-experiment` (missing evidence) | FATAL | FATAL + exit 1 | ✅ |
| unknown command | error | error + exit 2 | ✅ |

---

## 三、`03_claim_lint.py` 验证详情

### 3.1 检查覆盖

| 检查 | 类型 | 状态 |
|------|------|------|
| schema_version = 1 | ERROR | ✅ |
| 无重复 claim_id | ERROR | ✅ |
| revisions 非空 | ERROR | ✅ |
| revision 编号连续 | ERROR | ✅ |
| 唯一 live revision (superseded_by=null) | ERROR | ✅ |
| current_revision 指针正确 | ERROR | ✅ |
| lifecycle_status 枚举合法 | ERROR | ✅ |
| status 枚举合法 | ERROR | ✅ |
| change_type 枚举合法 | ERROR | ✅ |
| change_reason 必填（非 create） | ERROR | ✅ |
| superseded_by_reason 必填（有 supersession） | ERROR | ✅ |
| superseded_by 不能指向自身 | ERROR | ✅ |
| supersession chain 完整性 | ERROR | ✅ |
| superseded/split lifecycle 一致性 | WARNING | ✅ |
| contradicted/supported evidence 必填 | WARNING | ✅ |

### 3.2 当前 WARNING 解释（5 个，均为 informational）

| Warning | 说明 | 处理建议 |
|---------|------|---------|
| `C_DATASET_SUFFICIENT@1: status=contradicted but evidence_ids is empty` | 数据量 claim 建立时无实际数据支撑，符合预期 | 当前正确，E002/E003 等补充后更新 |
| `C_SIGLIP_GENERALIZATION@2: evidence_ids contains 'E002' but not in index` | 引用了尚未登记的 E002（E1 held-out eval） | 未来登记 E002 后此 WARNING 消失 |
| `C_SIGLIP_GENERALIZATION@3: evidence_ids contains 'E002' but not in index` | 同上 | 同上 |
| `C_SIGLIP_GENERALIZATION@3: evidence_ids contains 'E003' but not in index` | 引用了尚未登记的 E003（D1 V57 finetuned eval） | 未来登记 E003 后此 WARNING 消失 |
| `C_FINETUNE_IMPROVES_HELDOUT@1: evidence_ids contains 'E002' but not in index` | 同 E002 | 同上 |

**结论**：5 个 WARNING 均为预期状态（待登记的 E002/E003），不是错误。

---

## 四、`sovereign_cli.py` 验证详情

### 4.1 子命令矩阵

| 子命令 | 写 sovereign | 原子性 | 验证 | 状态 |
|--------|------------|--------|------|------|
| `bootstrap` | ❌ | — | 调用 3 个 lint | ✅ |
| `reconcile` | ❌ | — | 运行所有 lint | ✅ |
| `render-truth` | ✅ (CAMPAIGN_TRUTH) | ✅ rename | 验证 sovereign 数据可读 | ✅ |
| `revise-claim` | ✅ (claims.yaml) | ✅ rename | claim 存在 + lifecycle 检查 | ✅ |
| `supersede-claim` | ✅ (claims.yaml) | ✅ rename | old/new claim 存在性检查 | ✅ |
| `split-claim` | ✅ (claims.yaml) | ✅ rename | source 存在 + target 不冲突 | ✅ |
| `propose-revision` | ✅ (proposals/) | ✅ rename | 写 proposal，不写 claims | ✅ |
| `record-evidence` | ✅ (evidence/) | ✅ rename | 文件存在 + 字段验证 | ✅ |
| `close-experiment` | ✅ (state.json) | ✅ rename | evidence 已登记 | ✅ |
| `write-handoff` | ✅ (handoff.md) | ✅ rename | 无预检查 | ✅ |

### 4.2 Bootstrap 输出样本

```
=== sovereign_cli.py bootstrap ===
Campaign: mint_drawer_v1

  PASS: 02_reconcile_sources
  PASS: 03_claim_lint
  PASS: 04_action_lint

=== SESSION BOOTSTRAP ===
Sovereign version: 1
Active claims: 4
  C_SIGLIP_GENERALIZATION: rev3 active candidate (superseded 2 older)
    scope: current_mint_drawer_v1_pipeline
    statement: Current failure is more likely system-level misalignment...
  C_VQVAE_USABLE: rev1 active contradicted
  C_DATASET_SUFFICIENT: rev1 active contradicted
  C_FINETUNE_IMPROVES_HELDOUT: rev1 active contradicted

Recent claim changes:
  2026-03-31 19:00 C_SIGLIP_GENERALIZATION@3 [reframed]: ...
  2026-03-31 19:00 C_VQVAE_USABLE@1 [create]: ...
  2026-03-31 19:00 C_DATASET_SUFFICIENT@1 [create]: ...

Superseded claims: none
Lint warnings (6): informational only

=== PROCEED WITH ANALYSIS ===
```

---

## 五、回归检查（对 Phase 1 的保护）

| 检查项 | 结果 |
|--------|------|
| Phase 1 sovereign 文件全部未动 | ✅ |
| Phase 1 symlinks 保持正确 | ✅ |
| Phase 1 legacy markers 保持 | ✅ |
| Phase 1 `02_reconcile_sources.py` 行为不变 | ✅ |
| V58 pipeline 路径未触碰 | ✅ |
| `state.json` 格式未改动（JSON 格式保留） | ✅ |

---

## 六、Phase 3 待实现清单

| 待实现 | 状态 |
|--------|------|
| `gc_01_stale_docs.py`（>7 天 .md/.yaml/.json） | 待实现 |
| `gc_02_orphan_artifacts.py`（未登记 artifacts） | 待实现 |
| `gc_03_conflicting_claims.py`（含 revision chain 检查） | 待实现 |
| `gc_04_outdated_next_actions.py`（superseded/split-aware） | 待实现 |
| Night runner P3a（watcher + planner + handoff） | 待实现 |
| Night runner P3b（verifier + dry-run） | 待实现 |
| Night runner P3c（bounded mutation） | 待实现 |
