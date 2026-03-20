# Infinigen x PhysNAP Phase 2 Negative Result Root-Cause Tree

**Updated**: 2026-03-20T21:00:00+08:00  
**Phase**: `phase2_conditioning_design`  
**Verdict**: `claim_not_supported`  
**Strongest true claim**: `Mixed replay is the strongest validated transfer improvement, but richer conditioning under the current PhysNAP parameterization does not consistently beat the zero-state single-view anchor.`

## Executive Summary

The current negative result is best explained by a **conditioning-interface / architecture mismatch** inside PhysNAP's existing conditional guidance path, not by a raw Infinigen asset-generation failure.

Observed metric pattern:
- richer conditioning improves average fit slightly (`MMD` decreases)
- diversity does not improve and often worsens (`COV` stays flat or drops)
- physical consistency degrades, especially for `multistate_multiview` (`E_pen`, `E_mob` increase sharply)

This is the signature pattern:
`average-fit improved, diversity worsened, physical consistency broken`

Interpretation locked in for this run:
- richer observations are being generated and exported correctly
- but under the current PhysNAP interface they are flattened into a lossy single-cloud bottleneck
- the model is not consuming explicit state/view semantics
- therefore "more information" becomes "more mixed geometry through the same interface", which can help average fit while hurting diversity and physics

## Evidence Snapshot

| Group | MMD | COV | 1NN | E_pen | E_mob | Cond Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| zero_singleview | 0.5589997228235006 | 0.0908203125 | 0.9250000044703484 | 0.0007139240296964999 | 0.0005797943085781299 | 0.0022055536246625707 |
| zero_multiview | 0.5496429605409503 | 0.08984375 | 0.9241071473807096 | 0.0007759372820146382 | 0.0005648964070132934 | 0.0014058477245271206 |
| multistate_singleview | 0.5471300091594458 | 0.0908203125 | 0.9250000044703484 | 0.0022372949169948697 | 0.0006206683465279639 | 0.0013572014092157285 |
| multistate_multiview | 0.5368264261633158 | 0.0888671875 | 0.9214285761117935 | 0.0091311283952867 | 0.010760995797075642 | 0.0014382785884663463 |

## 1. 数据问题

### 已排除
- `shape code` 为零向量不是当前主因。此前已确认 codebook 为真实编码，而不是 placeholder zeros。
- 旧的 `visual origin` bridge blocker 不是当前 Phase 2 的主因。桥接修复后 GT self-eval 已恢复健康。
- repaired 之后的 Infinigen GT 并未表现出 catastrophic invalidity，因此不能把当前结果直接归因到原始 data engine 全坏。

### 尚未排除
- richer observations 在导出阶段仍可能引入 aliasing/noise，尤其是多个 state/view 被合并后形成的混合表面。
- 当前 observation coverage 仍可能在类别、状态幅度、可见面分布上不均衡。
- 当前点数预算固定为 1000，可能不足以无损承载多状态多视角信息。

### 当前最可能结论
上游数据不是主失败源；当前数据层最可能的问题是：**richer observations 在进入模型前被压缩成了 lossy merged cloud，信息保真不足。**

## 2. 接口问题

### 已排除
- richer conditioning datasets 并不缺失。四组 `cond_dir` 都已成功导出。
- 模型并非拿到了完全相同的原始输入；四组导出策略不同，输出数据确实发生变化。
- "没有 multi-state / multi-view observation on disk" 不是事实。

### 尚未排除
- merged multi-state / multi-view clouds 会把不同 articulation states 的表面点混进同一个点云，引入自冲突几何。
- state/view metadata 虽然写入了 sidecar，但 PhysNAP inference 并没有消费这些 metadata。
- 1000 点固定下采样会抹掉 richer observations 的结构优势，把“多证据”压成“更混杂的单证据”。

### 当前最可能主因
这是当前最强主因：**richer observations 被压平成单个无序 1000 点点云，而 state/view metadata 从未进入推理接口，因此“更多信息”在模型看来只是“更多混合几何穿过同一个瓶颈”。**

### 直接代码证据
- `scripts/build_conditioning_v2_dataset.py`
  - multiple states/views are concatenated into one merged point set
  - merged cloud is resampled back to a fixed point budget
- `external/physnap/eval/run_guided.py`
  - consumes `cond_dir/pcs.npy`
  - does not consume explicit state/view metadata

## 3. 架构问题

### 已排除
- pure converter failure 不是当前唯一解释；Phase 0/bridge repair 已经通过。
- pure catastrophic forgetting 不是当前 Phase 2 的唯一解释；Phase 2 base checkpoint 已经是 `mixed_finetune_k10`，不是最脆弱的 pure finetune 分支。

### 尚未排除
- PhysNAP 当前 guidance path 可能天然假设输入是 `single-observation, zero-articulation, limited-view`。
- architecture 可能不支持 cross-state conditioning，至少不支持在没有 set-aware encoder 或 explicit metadata fusion 的情况下支持它。
- guidance losses 可能默认输入对应一个 coherent observation state，因此在 mixed-state merged cloud 下出现 conflicting constraints。

### 当前最可能配对主因
**PhysNAP 当前 conditional architecture 与 multi-state / multi-view conditioning 不匹配。**

换句话说，不是 richer observations 没价值，而是当前模型接口没有合适的结构去利用这类 richer observations。

## 4. 指标问题

### 已排除
- 这次 Phase 2 最终 verdict 不是由 `invalid_phase2_summary` 之类的执行 bug 造成的。最终 `comparison_summary.json` / `comparison_report.md` / `decision_memo.md` 已经完整生成。
- 当前 negative result 不是单纯因为 missing summary files 或坏 `stats.json`。

### 尚未排除
- 当前 success rule 很严格：要同时满足 structural gain、physics preserved、conditioning preserved。
- 某些指标可能会惩罚 mixed-state ambiguity，即便其中存在部分有用语义。
- `COV` 与 physics metrics 是否对 richer merged observations 过于敏感，还没有通过专门 ablation 完全验证。

### 当前最可能结论
指标不是主失败源；它们更像是在揭露一个真实现象：
- average fit pressure 增强了，所以 `MMD` 略有改善
- 多样性没有同步变好，所以 `COV` 持平或下降
- articulation/physics consistency 变差，所以 `E_pen` / `E_mob` 上升

因此指标当前是在**暴露接口/架构不匹配的后果**，而不是单独制造一个假负结果。

## Final Root-Cause Ranking

1. **接口问题（最高）**  
   richer observations 被扁平化成单一点云，metadata 没进模型。

2. **架构问题（高）**  
   PhysNAP 当前 conditional guidance parameterization 更适合 `zero-state single-observation`，不适合 merged cross-state evidence。

3. **数据问题（中）**  
   不是 raw data engine 崩坏，但 merged observation construction 仍可能带来 aliasing 和 coverage 偏差。

4. **指标问题（低到中）**  
   需要后续继续验证其敏感性，但目前更像揭示真实 failure pattern，而不是主要根因。

## What This Run Does *Not* Support

This run does **not** support the statement:
- `multi-state + multi-view procedural articulated observations provide a stronger conditioning signal than PhysNAP's original zero-state single-view setting`

It also does **not** justify the stronger negative claim:
- `multi-state information has no value`

The more defensible conclusion is narrower:
- richer observations were generated,
- but the current PhysNAP conditioning interface and architecture did not exploit them effectively.
