"""Compare Paligemma LM weights: local checkpoint vs fresh HF initialization."""
import os, sys
from safetensors import safe_open
import torch

sys.path.insert(0, '/mnt/afs2/zhuhaowu/infinigen/external/MINT/lerobot_policy_mint/src')
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import json
from transformers import PaliGemmaConfig

cfg_path = '/mnt/afs2/zhuhaowu/infinigen/external/MINT/checkpoints/paligemma-3b-pt-224/config.json'
with open(cfg_path) as f:
    cfg = json.load(f)

# ============================================================
# METHOD 1: Load from LOCAL MINT checkpoint (via loaded model)
# ============================================================
from lerobot_policy_mint.modeling_mint import MINTPolicy
print("Loading MINT from local checkpoint...")
policy_local = MINTPolicy.from_pretrained(
    '/mnt/afs2/zhuhaowu/infinigen/external/MINT/checkpoints/MINT-libero',
    local_files_only=True
)
policy_local.eval()

pal_local = policy_local.model.paligemma_with_expert.paligemma
for attr in dir(pal_local):
    obj = getattr(pal_local, attr, None)
    if obj is not None and hasattr(obj, 'model') and 'language' in attr.lower():
        lm_attr = attr
        break

lm_inner = getattr(pal_local, lm_attr).model
local_embed = lm_inner.embed_tokens.weight.detach().cpu().float()
local_qproj0 = lm_inner.layers[0].self_attn.q_proj.weight.detach().cpu().float()
local_ln0 = lm_inner.layers[0].input_layernorm.weight.detach().cpu().float()

print(f"\n[LOCAL] Paligemma LM attr: {lm_attr}")
print(f"[LOCAL] embed_tokens: shape={tuple(local_embed.shape)}, std={local_embed.std().item():.6f}")
print(f"[LOCAL] q_proj[0]: std={local_qproj0.std().item():.6f}")
print(f"[LOCAL] ln[0] mean={local_ln0.mean().item():.6f}, std={local_ln0.std().item():.6f}")

# ============================================================
# METHOD 2: Fresh PaliGemma initialization (no pretrained weights)
# ============================================================
print("\nCreating FRESH PaliGemma (no pretrained weights)...")
vlm_config_hf = PaliGemmaConfig(
    vocab_size=cfg['text_config']['vocab_size'],
    hidden_size=cfg['text_config']['hidden_size'],
    intermediate_size=cfg['text_config']['intermediate_size'],
    num_hidden_layers=cfg['text_config']['num_hidden_layers'],
    num_attention_heads=cfg['text_config']['num_attention_heads'],
    num_key_value_heads=cfg['text_config']['num_key_value_heads'],
    image_token_index=cfg['image_token_index'],
    torch_dtype=torch.bfloat16,
)

from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
fresh_pali = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
fresh_pali.eval()

for attr in dir(fresh_pali):
    obj = getattr(fresh_pali, attr, None)
    if obj is not None and hasattr(obj, 'model') and 'language' in attr.lower():
        fresh_lm_attr = attr
        break

fresh_inner = getattr(fresh_pali, fresh_lm_attr).model
fresh_embed = fresh_inner.embed_tokens.weight.detach().cpu().float()
fresh_qproj0 = fresh_inner.layers[0].self_attn.q_proj.weight.detach().cpu().float()
fresh_ln0 = fresh_inner.layers[0].input_layernorm.weight.detach().cpu().float()

print(f"\n[FRESH] Paligemma LM attr: {fresh_lm_attr}")
print(f"[FRESH] embed_tokens: shape={tuple(fresh_embed.shape)}, std={fresh_embed.std().item():.6f}")
print(f"[FRESH] q_proj[0]: std={fresh_qproj0.std().item():.6f}")
print(f"[FRESH] ln[0] mean={fresh_ln0.mean().item():.6f}, std={fresh_ln0.std().item():.6f}")

# ============================================================
# PART 3: Load PaliGemma WITH pretrained weights (from_pretrained)
# ============================================================
hf_name = cfg.get('_name_or_path', '').replace('final-hf/', '')
print(f"\nLoading PaliGemma with pretrained weights from: {hf_name}")
try:
    hf_pali = PaliGemmaForConditionalGeneration.from_pretrained(
        hf_name,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
        local_files_only=False
    )
    hf_pali.eval()
    for attr in dir(hf_pali):
        obj = getattr(hf_pali, attr, None)
        if obj is not None and hasattr(obj, 'model') and 'language' in attr.lower():
            hf_lm_attr = attr
            break
    hf_inner = getattr(hf_pali, hf_lm_attr).model
    hf_embed = hf_inner.embed_tokens.weight.detach().cpu().float()
    hf_qproj0 = hf_inner.layers[0].self_attn.q_proj.weight.detach().cpu().float()
    hf_ln0 = hf_inner.layers[0].input_layernorm.weight.detach().cpu().float()
    hf_loaded = True
    print(f"[HF] embed_tokens: std={hf_embed.std().item():.6f}")
    print(f"[HF] q_proj[0]: std={hf_qproj0.std().item():.6f}")
    print(f"[HF] ln[0] mean={hf_ln0.mean():.6f}, std={hf_ln0.std().item():.6f}")
except Exception as e:
    hf_loaded = False
    print(f"[HF] Failed to load: {e}")

# ============================================================
# COMPARISONS
# ============================================================
print(f"\n{'='*60}")
print("COMPARISON 1: LOCAL MINT vs FRESH INIT")
min_v = min(local_embed.shape[0], fresh_embed.shape[0])
embed_mae_vs_fresh = (local_embed[:min_v] - fresh_embed[:min_v]).abs().mean().item()
q_mae_vs_fresh = (local_qproj0 - fresh_qproj0).abs().mean().item()
ln_mae_vs_fresh = (local_ln0 - fresh_ln0).abs().mean().item()
print(f"  embed_tokens MAE: {embed_mae_vs_fresh:.8f}")
print(f"  q_proj[0] MAE:    {q_mae_vs_fresh:.8f}")
print(f"  ln[0] gamma MAE:  {ln_mae_vs_fresh:.8f}")

if hf_loaded:
    print(f"\nCOMPARISON 2: LOCAL MINT vs HF PRETRAINED")
    min_v2 = min(local_embed.shape[0], hf_embed.shape[0])
    embed_mae_vs_hf = (local_embed[:min_v2] - hf_embed[:min_v2]).abs().mean().item()
    q_mae_vs_hf = (local_qproj0 - hf_qproj0).abs().mean().item()
    ln_mae_vs_hf = (local_ln0 - hf_ln0).abs().mean().item()
    print(f"  embed_tokens MAE: {embed_mae_vs_hf:.8f}")
    print(f"  q_proj[0] MAE:    {q_mae_vs_hf:.8f}")
    print(f"  ln[0] gamma MAE:  {ln_mae_vs_hf:.8f}")
print(f"{'='*60}")

THRESHOLD = 1e-4
if hf_loaded:
    if embed_mae_vs_hf < THRESHOLD and q_mae_vs_hf < THRESHOLD:
        print("\n✅ MINT Paligemma LM ≈ HuggingFace 预训练权重")
        print("   -> Paligemma LM 未被 MINT fine-tune（或退回了预训练）")
    else:
        print("\n⚠️  MINT Paligemma LM ≠ HuggingFace 预训练权重")
        print("   -> Paligemma LM 在 MINT 训练中被 fine-tune 过")

if embed_mae_vs_fresh < THRESHOLD and q_mae_vs_fresh < THRESHOLD:
    print("\n⚠️  MINT Paligemma LM ≈ 随机初始化")
    print("   -> Paligemma LM 没有加载任何权重（checkpoint 缺失）")
else:
    print("\n✅ MINT Paligemma LM ≠ 随机初始化")
    print("   -> Paligemma LM 加载了有效权重（来自某处）")
