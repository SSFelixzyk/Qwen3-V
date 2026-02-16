import torch
from transformers import AutoTokenizer
from model.model_vlm import Qwen3VLM, VLMConfig

llm_path = "../Models/Qwen3-0.6B"
vision_path = "./model/vision_model/siglip2-base-patch16-224"
delta_path = "./out/pretrain_vlm_512.pth"  # 或 ./checkpoints/<name>.pth

config = VLMConfig.from_pretrained(llm_path, trust_remote_code=True)
config.use_deepstack = False

model = Qwen3VLM(config=config, vision_model_path=vision_path)

state = torch.load(delta_path, map_location="cpu")
missing, unexpected = model.load_state_dict(state, strict=False)
print("missing:", [k for k in missing if "vision_encoder" not in k])  # vision_encoder 缺失是正常的
print("unexpected:", unexpected)

tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)

model = model.to("cuda").eval()
