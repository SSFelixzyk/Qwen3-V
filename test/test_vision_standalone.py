
import torch
from transformers import AutoModel, AutoProcessor
import os

def test_siglip_output():
    model_path = "./model/vision_model/siglip2-base-patch16-224"
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist.")
        return

    print(f"Loading model from {model_path}...")
    try:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
        print("Model loaded.")
        print(f"Model type: {type(model)}")
        
        # Check if it has .vision_model
        if hasattr(model, 'vision_model'):
            print("Model has .vision_model attribute.")
            print(f"Vision model type: {type(model.vision_model)}")
            # FORCE COMPUTE HIDDEN STATES in CONFIG
            model.vision_model.config.output_hidden_states = True
            print(f"Vision Config output_hidden_states: {model.vision_model.config.output_hidden_states}")
        else:
            print("Model DOES NOT have .vision_model attribute.")
            model.config.output_hidden_states = True
            print(f"Config output_hidden_states: {model.config.output_hidden_states}")

        # Create dummy input
        # SigLIP-Base usually expects 224x224
        dummy_image = torch.randn(1, 3, 224, 224)
        
        print("-" * 20)
        print("Attempt 1: Calling model.vision_model(...)")
        
        import transformers
        print(f"Transformers version: {transformers.__version__}")
        
        # Verify and force config on encoder
        if hasattr(model.vision_model, 'encoder'):
             print("model.vision_model.encoder found.")
             # Check its config
             if hasattr(model.vision_model.encoder, 'config'):
                  print(f"model.vision_model.encoder.config.output_hidden_states: {model.vision_model.encoder.config.output_hidden_states}")
                  model.vision_model.encoder.config.output_hidden_states = True
                  print(f"FORCED model.vision_model.encoder.config.output_hidden_states: {model.vision_model.encoder.config.output_hidden_states}")
             else:
                  print("model.vision_model.encoder has no 'config' attribute.")
                  
        # Also force again on top level
        model.vision_model.config.output_hidden_states = True

        with torch.no_grad():
            # Try passing as parameter to encoder
            # If not working, we'll try something else
            pass

        with torch.no_grad():
            if hasattr(model, 'vision_model'):
                # Try calling encoder DIRECTLY as we had issues with wrapper
                pass
            else:
                 pass
        
        # Breakdown again
        try:
             # 1. Embeddings
             print("Calling vision_model.embeddings...")
             hidden_states = model.vision_model.embeddings(pixel_values=dummy_image)
             print(f"Embeddings output shape: {hidden_states.shape}")
             
             import inspect
             print(f"Encoder forward signature: {inspect.signature(model.vision_model.encoder.forward)}")
             
             # 2. Encoder
             print("Calling vision_model.encoder with output_hidden_states=True...")
             # Force kwargs explicitly
             encoder_outputs = model.vision_model.encoder(
                 inputs_embeds=hidden_states,
                 output_hidden_states=True,
                 return_dict=True
             )
             print(f"Encoder Output type: {type(encoder_outputs)}")
             print(f"Encoder Output attributes: {dir(encoder_outputs)}")
             
             # Check if it's acting like a dict but hidden_states key is missing or None
             if isinstance(encoder_outputs, dict):
                 print(f"Encoder Output keys: {encoder_outputs.keys()}")
             elif hasattr(encoder_outputs, 'to_tuple'):
                 print(f"Encoder Output tuple form: {len(encoder_outputs.to_tuple())}")
             
             if hasattr(encoder_outputs, 'hidden_states'):
                 if encoder_outputs.hidden_states is None:
                     print("Encoder hidden_states is None!")
                 else:
                     print(f"Encoder hidden_states is present. Length: {len(encoder_outputs.hidden_states)}")
             elif hasattr(encoder_outputs, 'last_hidden_state'):
                 # Maybe it's just returning last_hidden_state?
                 print("Only last_hidden_state found.")
             
        except Exception as e_breakdown:
             print(f"Breakdown failed: {e_breakdown}")
             import traceback
             traceback.print_exc()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_siglip_output()
