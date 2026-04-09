"""Verify how Paligemma LM weights are loaded in MINT."""
import os, sys, torch
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
from transformers.models.auto import CONFIG_MAPPING

# Step 1: What does CONFIG_MAPPING["paligemma"]() return?
vlm_config = CONFIG_MAPPING["paligemma"]()
print(f"CONFIG_MAPPING['paligemma']() returns:")
print(f"  type: {type(vlm_config).__name__}")
print(f"  _name_or_path: {vlm_config._name_or_path}")
print(f"  text_config._name_or_path: {vlm_config.text_config._name_or_path}")

# Step 2: Load Paligemma from CONFIG_MAPPING config (like MINT does)
from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
print("\nLoading PaliGemma with CONFIG_MAPPING config (like MINT)...")
pali_from_config = PaliGemmaForConditionalGeneration(config=vlm_config)

for attr in dir(pali_from_config):
    obj = getattr(pali_from_config, attr, None)
    if obj is not None and hasattr(obj, 'model') and 'language' in attr.lower():
        lm_attr = attr
        break

lm_inner = getattr(pali_from_config, lm_attr).model
embed_w = lm_inner.embed_tokens.weight.detach().cpu().float()
qproj0_w = lm_inner.layers[0].self_attn.q_proj.weight.detach().cpu().float()
ln0_w = lm_inner.layers[0].input_layernorm.weight.detach().cpu().float()

print(f"\nembed_tokens: std={embed_w.std().item():.6f}")
print(f"q_proj[0]:   std={qproj0_w.std().item():.6f}")
print(f"ln[0]:       mean={ln0_w.mean().item():.6f}")

# Step 3: Compare with MINT loaded model
sys.path.insert(0, '/mnt/afs2/zhuhaowu/infinigen/external/MINT/lerobot_policy_mint/src')
from lerobot_policy_mint.modeling_mint import MINTPolicy
print("\nLoading MINT policy...")
policy = MINTPolicy.from_pretrained(
    '/mnt/afs2/zhuhaowu/infinigen/external/MINT/checkpoints/MINT-libero',
    local_files_only=True
)
policy.eval()

pal = policy.model.paligemma_with_expert.paligemma
for attr in dir(pal):
    obj = getattr(pal, attr, None)
    if obj is not None and hasattr(obj, 'model') and 'language' in attr.lower():
        lm_attr2 = attr
        break

lm2 = getattr(pal, lm_attr2).model
embed2 = lm2.embed_tokens.weight.detach().cpu().float()
qproj2 = lm2.layers[0].self_attn.q_proj.weight.detach().cpu().float()
ln2 = lm2.layers[0].input_layernorm.weight.detach().cpu().float()

print(f"\nembed_tokens: std={embed2.std().item():.6f}")
print(f"q_proj[0]:   std={qproj2.std().item():.6f}")
print(f"ln[0]:       mean={ln2.mean().item():.6f}")

# Compare
min_v = min(embed_w.shape[0], embed2.shape[0])
e_mae = (embed_w[:min_v] - embed2[:min_v]).abs().mean().item()
q_mae = (qproj0_w - qproj2).abs().mean().item()
ln_mae = (ln0_w - ln2).abs().mean().item()

print(f"\n{'='*50}")
print(f"embed_tokens MAE (from_config vs MINT): {e_mae:.8f}")
print(f"q_proj[0] MAE (from_config vs MINT):    {q_mae:.8f}")
print(f"ln[0] gamma MAE (from_config vs MINT):  {ln_mae:.8f}")
print(f"{'='*50}")

T = 1e-4
if e_mae < T and q_mae < T:
    print("\n✅ MINT Paligemma LM == CONFIG_MAPPING 加载的预训练权重")
    print("   -> Paligemma LM 权重来自 HuggingFace 的 google/paligemma-3b-pt-224")
    print("   -> 博士师兄说的 'base 里就包括了' 就是这个意思！")
    print("   -> Paligemma LM 未被 MINT fine-tune（训练时被冻结）")
else:
    print("\n⚠️  权重不同")
