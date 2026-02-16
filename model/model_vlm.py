import os
import torch
import warnings
from .model_minimind import *
from typing import Optional, Tuple, List, Union
from torch import nn
from transformers import AutoModel, AutoProcessor, Qwen3ForCausalLM, Qwen3Config
from transformers.modeling_outputs import CausalLMOutputWithPast

warnings.filterwarnings('ignore')


class VLMConfig(Qwen3Config):
    """
    VLM Configuration Class
    -----------------------
    Inherits from Qwen3Config to ensure compatibility with Qwen3's LLM structure.
    Adds VLM-specific parameters for vision encoder and Deepstack projector.
    """
    model_type = "Qwen3-v"

    def __init__(
            self,
            # Qwen3 token IDs: <|image_pad|> = 151655
            # We use 196 tokens as placeholder (14x14 patches)
            image_special_token: str = '<|image_pad|>' * 196, 
            image_ids: List = [151655] * 196, # List of token IDs representing the image placeholder
            vision_select_layer: int = -1, # Default to last layer (unused if deepstack is True)
            use_deepstack: bool = False,   # Whether to use Deepstack feature extraction
            # SigLIP-Base has 12 layers. We select last 4 even layers: 12, 10, 8, 6 (indices -1, -3, -5, -7 or similar)
            # Standard ViT features are often better from slightly earlier layers.
            # Let's use indices compatible with 12 layers: [-2, -5, -8, -11]
            deepstack_layers: List[int] = [-2, -5, -8, -11], # Layers to extract features from for Deepstack
            **kwargs,
    ):
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        self.vision_select_layer = vision_select_layer
        self.use_deepstack = use_deepstack
        self.deepstack_layers = deepstack_layers
        super().__init__(**kwargs)

class DeepstackProjector(nn.Module):
    """
    Deepstack Projector
    -------------------
    An MLP that projects concatenated vision features to LLM's hidden size.
    Structure: Linear -> GELU -> Linear
    """
    def __init__(self, input_hidden_size=768, output_hidden_size=1024):
        super().__init__()
        self.input_hidden_size = input_hidden_size
        self.output_hidden_size = output_hidden_size
        self.net = nn.Sequential(
            nn.Linear(input_hidden_size, output_hidden_size),
            nn.GELU(),
            nn.Linear(output_hidden_size, output_hidden_size)
        )

    def forward(self, image_features):
        """
        Forward pass for the projector.
        
        Args:
            image_features: Tensor of shape [batch_size, seq_len, input_hidden_size]
                            (e.g., [B, 196, 768 * num_layers])
                            
        Returns:
            Tensor of shape [batch_size, seq_len, output_hidden_size]
            (e.g., [B, 196, 1024] for Qwen3-0.6B)
        """
        return self.net(image_features)


class Qwen3VLM(Qwen3ForCausalLM):
    """
    Multi-modal Qwen3 Model
    -----------------------
    Integrates SigLIP2 vision encoder with Qwen3 LLM using a Deepstack Projector.
    Inherits from Qwen3ForCausalLM to utilize its text generation capabilities.
    """
    config_class = VLMConfig

    def __init__(self, config: VLMConfig = None, vision_model_path="./model/vision_model/siglip2-base-patch16-224"):
        # Initialize Qwen2/3 model (LLM Backbone)
        print("Initializing Qwen3 base model...")
        super().__init__(config)
        print("Qwen3 base model initialized.")
        
        # Initialize Vision Components with SigLip
        # vision_encoder: The loaded SigLIP model
        # processor: The SigLIP image processor
        print(f"Loading vision model from {vision_model_path}...")
        self.vision_encoder, self.processor = self.__class__.get_vision_model(vision_model_path)
        print("Vision components loaded.")
        
        # Determine input size for projector based on configuration
        # If Deepstack is used, input size = vision_hidden_size * num_selected_layers
        if getattr(config, 'use_deepstack', False):
            ve_hidden_size = 768 * len(config.deepstack_layers)
        else:
            ve_hidden_size = 768
            
        # Initialize the Projector
        print(f"Initializing DeepstackProjector with input_size={ve_hidden_size}...")
        self.vision_proj = DeepstackProjector(input_hidden_size=ve_hidden_size, output_hidden_size=config.hidden_size)
        print("Qwen3VLM init complete.")

        llm_dtype = self.model.embed_tokens.weight.dtype
        self.vision_proj = self.vision_proj.to(dtype=llm_dtype)
        self.vision_encoder = self.vision_encoder.to(dtype=llm_dtype)

    @staticmethod
    def get_vision_model(model_path: str):
        """
        Loads the Vision Encoder (SigLIP) and Processor.
        
        Args:
            model_path: Path to the pretrained vision model folder.
            
        Returns:
            model: The vision model (eval mode, frozen parameters).
            processor: The image processor.
        """
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        if not os.path.exists(model_path):
            print(f"Error: Vision model path {model_path} does not exist!")
            return None, None
        # Load SigLIP or compatible model using AutoModel
        print(f"Loading Vision Model from {model_path}...")
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        print("Vision Model loaded.")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        print("Processor loaded.")
        
        # Force the vision model (inner) to output hidden states
        # Critical fix: Some wrappers like SiglipModel don't propagate this from call arg by default?
        model.config.output_hidden_states = True
        if hasattr(model, 'vision_model'):
            model.vision_model.config.output_hidden_states = True
            
        # Freeze vision encoder parameters to prevent updating them during VLM training
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processor

    @staticmethod
    def image2tensor(image, processor):
        """
        Preprocesses a raw PIL image into a model-compatible tensor.
        
        Args:
            image: PIL Image object.
            processor: The image processor.
            
        Returns:
            inputs: Tensor of shape [1, channels, height, width] 
                   (e.g., [1, 3, 224, 224] for SigLIP-Base)
        """
        if image.mode in ['RGBA', 'LA']: image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")['pixel_values']
        return inputs

    @staticmethod
    def get_image_embeddings(image_tensors, vision_model, config=None):
        """
        Extract image patch embeddings from SigLIP vision tower.

        Returns:
            img_embedding: [B, 196, 768] (for siglip patch16-224)
        """
        with torch.no_grad():
            # 1) If this is SiglipModel (dual-encoder), use its vision tower
            vision = vision_model.vision_model if hasattr(vision_model, "vision_model") else vision_model

            # 2) Call the standard forward; this matches what you just debugged
            out = vision(
                pixel_values=image_tensors,
                output_hidden_states=bool(config and getattr(config, "use_deepstack", False)),
                return_dict=True,
            )

            last = out.last_hidden_state  # [B, 196 or 197, 768] depending on model; yours is [B,196,768]

            # 3) If a model variant includes CLS, drop it (your current one doesn't, but keep it robust)
            if last.shape[1] == 197:
                last = last[:, 1:, :]

            # 4) Sanity check against your placeholder design (196 tokens)
            if config is not None and hasattr(config, "image_ids"):
                expected = len(config.image_ids)
                if last.shape[1] != expected:
                    raise ValueError(f"Vision tokens={last.shape[1]} but expected={expected} (check CLS/patch/resolution).")

            # 5) Deepstack (optional). Only works if hidden_states is returned.
            if config and getattr(config, "use_deepstack", False):
                hs = getattr(out, "hidden_states", None)
                if hs is None:
                    # Fallback: can't deepstack, return last
                    return last

                selected = []
                for layer_idx in getattr(config, "deepstack_layers", [-2, -5, -8, -11]):
                    if -len(hs) <= layer_idx < len(hs):
                        x = hs[layer_idx]
                        if x.shape[1] == 197:
                            x = x[:, 1:, :]
                        selected.append(x)
                return torch.cat(selected, dim=-1) if selected else last

            # 6) Standard path
            return last

    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
        """
        Injects projected vision features into the LLM's text embeddings.
        This function locates the special image placeholder tokens in the input sequence
        and replaces their corresponding embeddings with the projected vision features.
        
        Args:
            tokens: Input token IDs. Shape: [batch_size, seq_len]
            h: Text embeddings from LLM. Shape: [batch_size, seq_len, hidden_size]
            vision_tensors: Vision features. Shape: [batch_size, num_images, seq_len_vis, feature_dim]
            seqlen: Max sequence length for truncation.
            
        Returns:
            Modified embeddings with vision features injected. Shape: [batch_size, seq_len, hidden_size]
        """
        def find_indices(tokens, image_ids):
            """Finds start and end indices of the image placeholder tokens."""
            image_ids_tensor = torch.tensor(image_ids).to(tokens.device)
            len_image_ids = len(image_ids)
            if len_image_ids > tokens.size(1):
                return None
            tokens_view = tokens.unfold(1, len_image_ids, 1)
            matches = (tokens_view == image_ids_tensor).all(dim=2)
            return {
                batch_idx: [(idx.item(), idx.item() + len_image_ids - 1) for idx in
                            matches[batch_idx].nonzero(as_tuple=True)[0]]
                for batch_idx in range(tokens.size(0)) if matches[batch_idx].any()
            } or None

        image_indices = find_indices(tokens, self.config.image_ids) # {batch_idx: [(start_idx, end_idx), ...]}
        if vision_tensors is not None and image_indices:
            # Project vision features to LLM hidden size
            # vision_proj: [batch_size, num_images, seq_len_vis, llm_hidden_size]
            vision_proj = self.vision_proj(vision_tensors)
            
            if len(vision_proj.shape) == 3:
                vision_proj = vision_proj.unsqueeze(0) # [1, num_images, seq_len_vis, llm_hidden_size]
            
            new_h = []
            for i in range(h.size(0)):
                if i in image_indices:
                    h_i = h[i] # [seq_len, hidden_size]
                    img_idx = 0
                    # Replace placeholder embeddings with vision embeddings
                    for start_idx, end_idx in image_indices[i]:
                        if img_idx < vision_proj.size(1):
                            # Concatenate: [pre_img, vision_feat, post_img]
                            # Dimensions:
                            # h_i[:start_idx]: [start_idx, hidden_size]
                            # vision_proj[i][img_idx]: [seq_len_vis, hidden_size]
                            # h_i[end_idx + 1:]: [remaining, hidden_size]
                            h_i = torch.cat((h_i[:start_idx], vision_proj[i][img_idx], h_i[end_idx + 1:]), dim=0)[
                                  :seqlen]
                            img_idx += 1
                    new_h.append(h_i)
                else:
                    new_h.append(h[i])
            return torch.stack(new_h, dim=0) # [batch_size, seq_len, hidden_size]
        return h

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                use_cache: bool = None,
                output_attentions: bool = None,
                output_hidden_states: bool = None,
                return_dict: bool = None,
                pixel_values: Optional[torch.FloatTensor] = None,
                **args):
        """
        Forward pass of the VLM.
        
        Args:
            input_ids: Token IDs. Shape: [batch_size, seq_len]
            pixel_values: Input images. 
                          Shape: [batch_size, num_images, channels, height, width] 
                          or [batch_size, channels, height, width] (single image)
            ... (other standard Qwen args)
            
        Returns:
            CausalLMOutputWithPast: Object containing loss, logits, and other outputs.
        """
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # If inputs_embeds not provided, compute from input_ids (Text Embeddings)
        # inputs_embeds: [batch_size, seq_len, hidden_size]
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        # Inject Vision Features if images are provided
        if pixel_values is not None:
            # Handle potential 5D input (b, n_images, c, h, w) or 4D (b, c, h, w)
            vision_device = next(self.vision_encoder.parameters()).device
            pixel_values = pixel_values.to(vision_device)

            
            # Case 1: 5D input [batch, num_images, channels, height, width]
            if len(pixel_values.shape) == 5: 
                 bs, num, c, im_h, im_w = pixel_values.shape
                 # Flatten to [batch * num_images, c, h, w] for efficient processing
                 pixel_values_flat = pixel_values.view(bs * num, c, im_h, im_w)
                 
                 # Get embeddings for all images: [batch*num, seq_len_vis, feature_dim]
                 vision_tensors_flat = Qwen3VLM.get_image_embeddings(pixel_values_flat, self.vision_encoder, self.config)
                 
                 # Reshape back to [batch, num, seq_len_vis, feature_dim]
                 seq_len_vis = vision_tensors_flat.shape[1]
                 feature_dim = vision_tensors_flat.shape[2]
                 vision_tensors = vision_tensors_flat.view(bs, num, seq_len_vis, feature_dim)
            
            # Case 2: 6D input (legacy/edge case) - Remove extra dim
            elif len(pixel_values.shape) == 6:
                pixel_values = pixel_values.squeeze(2) 
                # (Then process as Case 1 or 3 depending on shape, but simplified here)
                bs, num, c, im_h, im_w = pixel_values.shape
                pixel_values_flat = pixel_values.view(bs * num, c, im_h, im_w)
                vision_tensors_flat = Qwen3VLM.get_image_embeddings(pixel_values_flat, self.vision_encoder, self.config)
                vision_tensors = vision_tensors_flat.view(bs, num, -1, vision_tensors_flat.shape[-1])

            # Case 3: 4D input [batch, c, h, w] (Single image per sample)
            elif len(pixel_values.shape) == 4:
                 vision_tensors = Qwen3VLM.get_image_embeddings(pixel_values, self.vision_encoder, self.config)
                 # Add num_images dim: [batch, 1, seq_len_vis, feature_dim]
                 vision_tensors = vision_tensors.unsqueeze(1)
            
            else:
                raise ValueError(f"Unexpected pixel_values shape: {pixel_values.shape}")


            # Perform Feature Injection (Replace text placeholders with image features)
            # inputs_embeds will be modified in-place or replaced
            inputs_embeds = self.count_vision_proj(tokens=input_ids, h=inputs_embeds, vision_tensors=vision_tensors, seqlen=input_ids.shape[1])

        # Standard Qwen2/3 forward pass with modified embeddings
        return super().forward(
            input_ids=None, # We provide inputs_embeds instead
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
