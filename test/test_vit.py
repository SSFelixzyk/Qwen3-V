import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image
import numpy as np

VISION_PATH = "./model/vision_model/siglip2-base-patch16-224"  # 改成你的路径

def main():
    print("=" * 60)
    print("Loading model & processor...")
    print("=" * 60)

    model = AutoModel.from_pretrained(VISION_PATH, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(VISION_PATH, trust_remote_code=True)

    model.eval()

    print("Loaded model class:", model.__class__)
    if hasattr(model, "vision_model"):
        print("Detected dual-encoder SiglipModel. Using model.vision_model only.")
        vision = model.vision_model
    else:
        print("Model has no .vision_model. Treating as a pure vision model.")
        vision = model

    vision.eval()

    # ---- dummy image ----
    dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(dummy)

    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    print("pixel_values:", pixel_values.shape)

    print("=" * 60)
    print("Running vision forward...")
    print("=" * 60)

    with torch.no_grad():
        out = vision(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

    last = out.last_hidden_state
    hs = out.hidden_states

    print("last_hidden_state shape:", tuple(last.shape))
    print("num hidden_states:", len(hs) if hs is not None else None)
    if hs is not None:
        print("hidden_states[0] shape:", tuple(hs[0].shape))
        print("hidden_states[-1] shape:", tuple(hs[-1].shape))

    # ---- CLS vs patches check ----
    seq_len = last.shape[1]
    print("\nSequence length:", seq_len)
    if seq_len == 197:
        print("Likely includes CLS token: 1 + 14*14 patches")
        patch = last[:, 1:, :]   # drop CLS
        print("patch_only shape:", tuple(patch.shape), "(should be [B, 196, H])")
    elif seq_len == 196:
        print("Patch-only output (14*14 patches).")
    else:
        print("Unexpected seq_len. Check image size / patch size / model variant.")

if __name__ == "__main__":
    main()

