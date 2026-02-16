
import sys
import os
import torch
from transformers import AutoModel, AutoProcessor, AutoConfig

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_vlm import Qwen3VLM, VLMConfig, DeepstackProjector

def test_deepstack_vlm():
    print("="*50)
    print("Testing SigLIP2 + Deepstack + Qwen3 VLM Implementation")
    print("="*50)

    # 1. Config Test
    print("\n[1] Testing VLMConfig...")
    deepstack_layers = [-2, -8, -14, -20]
    config = VLMConfig(
        hidden_size=1024, # Qwen3-0.6B
        use_deepstack=True,
        deepstack_layers=deepstack_layers
    )
    print(f"✓ Config created: use_deepstack={config.use_deepstack}, layers={config.deepstack_layers}")
    
    # 2. Vision Model Loading Test (Mocking if files don't exist, but trying to use what's there)
    # Since specific SigLIP weights might not be present, we might need to mock or skip if not found.
    # However, for architecture test, we can mock the get_vision_model method if needed, 
    # but let's try to instantiate MiniMindVLM.
    
    # Hack: If model path doesn't exist, we can't test fully without downloading.
    # But we can test the class structure.
    
    print("\n[2] Testing Qwen3VLM Instantiation...")
    # We will mock get_vision_model to return a simple storage object if real one isn't there
    
    class MockVisionModel:
        def __init__(self):
            self.config = AutoConfig.from_pretrained('./model/vision_model/siglip2-base-patch16-224') # just for config if needed
            self.config.hidden_size = 768 
            
        def vision_model(self, pixel_values, output_hidden_states=True):
            batch_size = pixel_values.shape[0]
            # Mock outputs
            seq_len = 197 
            hidden_dim = 768
            
            # create mock hidden states
            # We need at least 25 layers to support -24 etc.
            valid_layers = 27 
            hidden_states = tuple(
                torch.randn(batch_size, seq_len, hidden_dim) for _ in range(valid_layers)
            )
            
            class Output:
                pass
            o = Output()
            o.hidden_states = hidden_states
            o.last_hidden_state = hidden_states[-1]
            return o
    
    class MockProcessor:
        def __call__(self, images, return_tensors='pt'):
            return {'pixel_values': torch.randn(1, 3, 224, 224)}

    # Patch the get_vision_model temporarily for testing locally if weights are missing
    original_get_vision = Qwen3VLM.get_vision_model
    
    # Check if path exists
    vision_path = "./model/vision_model/siglip2-base-patch16-224"
    if not os.path.exists(vision_path):
        print("  ! Vision model path not found, using Mock for architecture test.")
        Qwen3VLM.get_vision_model = lambda x: (MockVisionModel(), MockProcessor())
    
    try:
        model = Qwen3VLM(config, vision_model_path=vision_path)
        print("✓ Model instantiated successfully")
        
        # 3. Test Projector Dimensions
        print("\n[3] Testing Projector Dimensions...")
        assert isinstance(model.vision_proj, DeepstackProjector)
        expected_input_dim = 768 * len(deepstack_layers)
        print(f"  Expected input dim: {expected_input_dim}")
        print(f"  Actual input dim: {model.vision_proj.input_hidden_size}")
        assert model.vision_proj.input_hidden_size == expected_input_dim
        assert model.vision_proj.output_hidden_size == 1024
        print("✓ Projector dimensions correct")
        
        # 4. Test get_image_embeddings with Deepstack
        print("\n[4] Testing Deepstack Feature Extraction...")
        dummy_image_tensor = torch.randn(2, 3, 224, 224) # batch=2
        
        emb = Qwen3VLM.get_image_embeddings(dummy_image_tensor, model.vision_encoder, config)
        print(f"  Output shape: {emb.shape}")
        
        expected_shape = (2, 197, expected_input_dim)
        assert emb.shape == expected_shape
        print("✓ Deepstack extraction shape correct")
        
        # 5. Test Forward (Architecture check)
        print("\n[5] Testing Full Forward Pass (Architecture)...")
        # We need to mock Qwen2 internals since we didn't load weights and might not have them
        # Just checking if it crashes on shape mismatch in count_vision_proj
        
        # Mock input
        input_ids = torch.randint(0, 1000, (2, 10))
        # Embeddings mock
        with torch.no_grad():
             # call count_vision_proj directly
             # embedding shape: [2, 10, 1024]
             hidden_states = torch.randn(2, 10, 1024)
             
             # vision_tensors: [2, 1, 729, expected_dim] (1 image per sample)
             fake_vision_features = torch.randn(2, 1, 197, expected_input_dim)
             
             # Need valid image_ids in tokens
             # update input_ids to have image token (id 34 default)
             # image_ids is list [34]*196. But here let's simplify for test.
             # We need to match what VLMConfig has. 
             # Default VLMConfig has image_ids length 196. 
             # SigLIP has 729 tokens. We need to update image_ids in config or params.
             
             # Update config image_ids for this test to match SigLIP length
             model.params.image_ids = [34] * 197
             
             # Insert image tokens into input_ids
             # [tok, tok, img...img, tok]
             # inputs: 10 tokens. let's make it larger
             input_ids = torch.full((2, 800), 0)
             input_ids[:, 10:10+729] = 34
             
             hidden_states = torch.randn(2, 800, 1024)
             
             # Run count_vision_proj
             out_h = model.count_vision_proj(input_ids, hidden_states, fake_vision_features, seqlen=800)
             print(f"  Proj output shape: {out_h.shape}")
             assert out_h.shape == (2, 800, 1024)
             print("✓ count_vision_proj architecture check passed")

    except Exception as e:
        print(f"✗ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Restore original method
        Qwen3VLM.get_vision_model = original_get_vision

if __name__ == "__main__":
    test_deepstack_vlm()
