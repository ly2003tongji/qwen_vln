import os, sys, faulthandler
faulthandler.enable()
import torch, transformers
sys.path.insert(0, '/mnt/cpfs/prediction/lyyy/myself/VLN/StreamVLN')
sys.path.insert(0, '/mnt/cpfs/prediction/lyyy/myself/VLN/StreamVLN/streamvln')
from model.stream_video_vln import StreamVLNForCausalLM

mp = "/mnt/cpfs/prediction/lyyy/myself/VLN/StreamVLN/checkpoints/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"
attn = os.environ.get("ATTN", "flash_attention_2")
print("attn_impl =", attn)
cfg = transformers.AutoConfig.from_pretrained(mp)
model = StreamVLNForCausalLM.from_pretrained(mp, attn_implementation=attn, torch_dtype=torch.bfloat16, config=cfg, low_cpu_mem_usage=False)
model.to(0).eval()
tok = transformers.AutoTokenizer.from_pretrained(mp)
ids = tok("You are an autonomous navigation assistant. Move forward.", return_tensors="pt").input_ids.to(0)
emb = model.get_model().embed_tokens(ids)
print("emb", emb.shape, emb.dtype)
with torch.no_grad():
    out = model.model(inputs_embeds=emb, use_cache=True)
torch.cuda.synchronize()
print("LLM_FORWARD_OK", out[0].shape)
