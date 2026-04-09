# MuJoCo Pilot Phase 2 执行计划

## 背景与新发现

Codex 完成了 MuJoCo Pilot 6 个文件的代码和执行，P0 验证发现了最深层的根因：

```
165 LM keys 缺失（embed_tokens + 27层 Q/K/V/O + norms）
  → Language Model 随机初始化
  → State/Text conditioning 通道失效
  → Policy 实际是 image-only + VQ-VAE prior
  → Drawer 任务需要精确 state tracking
  → Gripper=-1, translation≈0 (VQ-VAE fallback)
  → 抽屉不动 = 0%
```

**新科学叙事**：`p1c4 = 0%` 不是「MINT 模型能力差」，而是「MINT 当前 inference pipeline 不包含有效的 state conditioning」。这是可修复的工程问题。

---

## 修正后的执行顺序

### Step 1 (P0): MINT LM 权重审计

**目标**：确认 165 missing keys 是「冻结随机初始化（设计选择）」还是「checkpoint 保存 bug」。

**执行**：

1. 检查 SDAT 训练配置中是否有 LM 冻结设置
2. 检查 `modeling_mint.py` 中 `embed_tokens` 是否从未被赋予有意义的权重
3. 如果 embed_tokens=None 贯穿始终 → 冻结随机（设计选择）

**结果判断**：

| 如果... | 说明 | 下一步 |
|--------|------|--------|
| LM 冻结随机 | 设计选择，非 bug | 重新设计 MINT inference 以 visual-only 模式运行 |
| LM 应该有权重 | checkpoint bug | 联系作者或从备份恢复 |

**输出**：`artifacts/p1cx_lm_weights_audit.json` + E033.yaml

---

### Step 2 (P1): LIBERO Action Oracle（Action-Space 版本）

**目标**：用真实的机器人动作（IK 控制器）证明 LIBERO drawer 任务可达。

**方法**：使用 LIBERO 的 robosuite 底层控制器，以初始化状态为起点，驱动机械臂向 handle 移动 → 夹爪闭合 → 拉抽屉。

**接受标准**：`success_rate >= 2/3`

**输出**：`artifacts/p1cx_libero_action_oracle.json` + E034.yaml

---

### Step 3 (P2): MINT Inference 设计审查

**目标**：确认 MINT 官方评测 LIBERO 时使用的 inference pipeline 与本地方案的差异。

**方法**：查找 MINT 官方评测代码，对比 `select_action` 调用差异。特别关注 warm-up steps、action queue 使用、`past_key_values` 重置。

**输出**：`artifacts/p1cx_mint_inference_review.json` + E035.yaml

---

### Step 4 (P3): State Mapping 验证

**目标**：用 QUANTILES ground truth stats 确定正确的 motor joint mapping。

**关键参考**：checkpoint 中 `observation.state` QUANTILES（第 3-7 维）：
```
dim[3]: mean=2.972, std=0.344
dim[4]: mean=-0.220, std=0.907
dim[5]: mean=-0.126, std=0.325
dim[6]: mean=+0.027, std=0.014
```

**方法**：从 LIBERO `OffScreenRenderEnv` 收集 200+ 帧 live joint 数据，对比不同 mapping 输出与 QUANTILES 的匹配度。

**输出**：`artifacts/p1cx_state_mapping.json` + E036.yaml

---

### Step 5 (P4): 干净的 MINT on LIBERO Drawer Baseline

**前置条件**：Step 1 和 Step 2 有明确结论。

**Pipeline 修正**：
- Task: `open_the_middle_drawer_of_the_cabinet`（libero_goal）
- Image size: 256×256（来自 policy_preprocessor.json）
- State mapping: 来自 Step 3 的正确映射
- Task text: 来自 LIBERO task.language 字段

**输出**：`artifacts/p1cx_mint_baseline_clean.json` + E037.yaml

---

## Night Runner Bundle

将 Step 1-4 打包为 `run_mujoco_pilot_phase2.py`，挂 screen 执行。

---

## 依赖关系

```
Step 1 (LM audit)
  ↓ 确定方向
Step 2 (Action oracle) ← 并行
Step 3 (Inference review) ← 并行
Step 4 (State mapping) ← 可并行
  ↓
Step 5 (Clean baseline) — 等 Step 1 有结论
```

Step 2-4 可并行执行，不需要等待 Step 1。Step 5 必须在 Step 1 有明确方向后才执行。
