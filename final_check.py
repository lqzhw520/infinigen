"""Final analysis: MINT checkpoint vs fresh vs pretrained Paligemma LM."""
import os, sys, torch
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
sys.path.insert(0, '/mnt/afs2/zhuhaowu/infinigen/external/MINT/lerobot_policy_mint/src')

# Load MINT policy
from lerobot_policy_mint.modeling_mint import MINTPolicy
policy = MINTPolicy.from_pretrained(
    '/mnt/afs2/zhuhaowu/infinigen/external/MINT/checkpoints/MINT-libero',
    local_files_only=True
)
policy.eval()

# Get Paligemma LM from loaded model
pal = policy.model.paligemma_with_expert.paligemma
for attr in dir(pal):
    obj = getattr(pal, attr, None)
    if obj is not None and hasattr(obj, 'model') and 'language' in attr.lower():
        lm_attr = attr
        break

lm_inner = getattr(pal, lm_attr).model
mint_embed = lm_inner.embed_tokens.weight.detach().cpu().float()
mint_qproj0 = lm_inner.layers[0].self_attn.q_proj.weight.detach().cpu().float()
mint_ln0 = lm_inner.layers[0].input_layernorm.weight.detach().cpu().float()

print("=== MINT Loaded Paligemma LM ===")
print(f"embed_tokens: shape={tuple(mint_embed.shape)}, std={mint_embed.std().item():.6f}")
print(f"q_proj[0]:    std={mint_qproj0.std().item():.6f}")
print(f"ln[0]:        mean={mint_ln0.mean().item():.6f}, std={mint_ln0.std().item():.6f}")

# Create FRESH (random init) Paligemma
from transformers import PaliGemmaConfig
from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
import json

with open('/mnt/afs2/zhuhaowu/infinigen/external/MINT/checkpoints/paligemma-3b-pt-224/config.json') as f:
    cfg = json.load(f)

vlm_config = PaliGemmaConfig(
    vocab_size=cfg['text_config']['vocab_size'],
    hidden_size=cfg['text_config']['hidden_size'],
    intermediate_size=cfg['text_config']['intermediate_size'],
    num_hidden_layers=cfg['text_config']['num_hidden_layers'],
    num_attention_heads=cfg['text_config']['num_attention_heads'],
    num_key_value_heads=cfg['text_config']['num_key_value_heads'],
    image_token_index=cfg['image_token_index'],
    torch_dtype=torch.bfloat16,
)

fresh_pali = PaliGemmaForConditionalGeneration(config=vlm_config)
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

print("\n=== Fresh Init Paligemma LM ===")
print(f"embed_tokens: shape={tuple(fresh_embed.shape)}, std={fresh_embed.std().item():.6f}")
print(f"q_proj[0]:    std={fresh_qproj0.std().item():.6f}")
print(f"ln[0]:        mean={fresh_ln0.mean().item():.6f}, std={fresh_ln0.std().item():.6f}")

# Compare
print("\n=== MINT vs Fresh Init ===")
min_v = min(mint_embed.shape[0], fresh_embed.shape[0])
e_mae = (mint_embed[:min_v] - fresh_embed[:min_v]).abs().mean().item()
q_mae = (mint_qproj0 - fresh_qproj0).abs().mean().item()
ln_mae = (mint_ln0 - fresh_ln0).abs().mean().item()
print(f"embed_tokens MAE: {e_mae:.8f}")
print(f"q_proj[0] MAE:    {q_mae:.8f}")
print(f"ln[0] gamma MAE:  {ln_mae:.8f}")

T = 1e-4
if e_mae < T and q_mae < T:
    print("\nRESULT: MINT Paligemma LM ≈ 随机初始化")
    print("  -> Paligemma LM 未从 checkpoint 加载权重（missing keys 的后果）")
else:
    print("\nRESULT: MINT Paligemma LM ≠ 随机初始化")
    print("  -> Paligemma LM 加载了有效权重（来自 from_pretrained）")
