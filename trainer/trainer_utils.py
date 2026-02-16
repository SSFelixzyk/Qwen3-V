"""
训练工具函数集合
Training Utility Functions
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from model.model_vlm import Qwen3VLM


def get_model_params(model, config, ignore_patterns=['vision_encoder']):
    """
    Calculates and logs the number of trainable parameters in the model.
    Ignores parameters matching 'ignore_patterns' (e.g., frozen vision encoder).
    """
    def should_count(n): return not any(p in n for p in ignore_patterns)
    total = sum(p.numel() for n, p in model.named_parameters() if should_count(n)) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n and should_count(n)) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n and should_count(n)) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    """Checks if the current process is the main process (rank 0) in distributed training."""
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    """Logs content to console only on the main process."""
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    """
    Calculates learning rate using Cosine Annealing schedule.
    Args:
        current_step: Current training step.
        total_steps: Total training steps.
        lr: Initial learning rate.
    Returns:
        Current learning rate.
    """
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    """
    Initializes distributed training (DDP) if applicable.
    Sets CUDA device based on local rank.
    """
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # Non-DDP mode
    
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_vlm_model(vlm_config, from_weight='../Models/Qwen3-0.6B', tokenizer_path='../Models/Qwen3-0.6B', 
                   vision_model_path='./model/vision_model/siglip2-base-patch16-224', 
                   save_dir='./out', device='cuda', freeze_llm=False, train_last_layer=False, enable_thinking=False):
    """
    Initializes the VLM model, tokenizer, and processor.
    
    Args:
        vlm_config: Configuration object for VLM.
        from_weight: Path or name of weights to load (e.g., '../Models/Qwen3-0.6B' or path to Qwen3).
        tokenizer_path: Path to the tokenizer (sibling to project root).
        vision_model_path: Path to the vision model (relative to project root).
        freeze_llm: Boolean, whether to freeze LLM parameters (for stage 2 pretraining).
        train_last_layer: Boolean, whether to train only the last layer of the LLM.
        enable_thinking: Boolean, whether to enable thinking mode.
    Returns:
        model: Initialized Qwen3VLM model.
        tokenizer: Loaded tokenizer.
        preprocess: Image processor from the vision model.
    """
    # Ensure usage of Qwen2/3 fast tokenizer if applicable, or fallback to auto
    Logger(f"Loading tokenizer from {tokenizer_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, enable_thinking=enable_thinking)
    except:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, trust_remote_code=True, enable_thinking=enable_thinking)
    Logger("Tokenizer loaded.")

    Logger(f"Initializing Qwen3VLM with vision_model_path={vision_model_path}...")
    model = Qwen3VLM(vlm_config, vision_model_path=vision_model_path)
    Logger("Qwen3VLM initialized.")
    
    if from_weight != 'none':
        Logger(f"Loading weights from {from_weight}...")
        # Check if from_weight is a path to Qwen3 weights (folder or safetensors)
        if os.path.isdir(from_weight):
            # Try loading matching weights (skip vision_proj)
            try:
                from safetensors import safe_open
                weight_file = os.path.join(from_weight, "model.safetensors")
                if os.path.exists(weight_file):
                    state_dict = {}
                    with safe_open(weight_file, framework="pt", device="cpu") as f:
                         for key in f.keys():
                             state_dict[key] = f.get_tensor(key)
                    
                    # Load weights into the model (non-strict to allow for VLM extensions)
                    model.load_state_dict(state_dict, strict=False)
                    Logger(f"Loaded Qwen3 weights from {weight_file}")
                else:
                    Logger(f"No model.safetensors found in {from_weight}")
            except Exception as e:
                Logger(f"Failed to load Qwen3 weights: {e}")

        # Try loading from a specific checkpoint name in save_dir
        moe_suffix = '_moe' if vlm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{vlm_config.hidden_size}{moe_suffix}.pth'
        if os.path.exists(weight_path):
            weights = torch.load(weight_path, map_location=device)
            model.load_state_dict(weights, strict=False)
        elif not os.path.isdir(from_weight):
             # Only warn if it wasn't a directory load attempt
             Logger(f"Warning: Checkpoint {weight_path} not found. Loading init weights.")
    
    Logger("Freezing LLM parameters..." if freeze_llm else "LLM parameters not frozen.")
    # Pretrain Stage: Freeze all parameters except vision_proj if freeze_llm is True
    if freeze_llm:
        for name, param in model.named_parameters():
            if 'vision_proj' not in name:
                param.requires_grad = False
    
    # Pretrain Stage: Unfreeze the last layer of LLM (optional, depending on strategy)
    if train_last_layer:
        last_layer_idx = vlm_config.num_hidden_layers - 1
        for name, param in model.model.named_parameters():
            if f'layers.{last_layer_idx}.' in name:
                param.requires_grad = True

    get_model_params(model, vlm_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    Logger(f'Trainable Params: {trainable_params:.3f}M')
    preprocess = model.processor
    return model.to(device), tokenizer, preprocess


def vlm_checkpoint(vlm_config, weight='pretrain_vlm', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='./checkpoints', **kwargs):
    """
    Saves or loads a checkpoint state.
    
    Args:
        model: If provided, saves the model state.
        optimizer: If provided, saves the optimizer state.
        save_dir: Directory to save checkpoints (defaults to ./checkpoints).
        kwargs: Additional states to save/load.
        
    Returns:
        checkpoint_data if loading, otherwise None.
    """
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if vlm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{vlm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{vlm_config.hidden_size}{moe_path}_resume.pth'
    
    if model is not None:
        # Saving Checkpoint
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        
        # Remove vision_encoder parameters (frozen, so no need to save)
        clean_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('vision_encoder.')}
        
        ckp_tmp = ckp_path + '.tmp'
        torch.save({k: v.half().cpu() for k, v in clean_state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)
        
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value
        
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, clean_state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # Loading Checkpoint
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU count changed ({saved_ws}→{current_ws}), step adjusted to {ckp_data["step"]}')
            return ckp_data
        return None


class SkipBatchSampler(Sampler):
    """
    Custom Sampler that skips a specified number of batches.
    Used for resuming training from a specific step within an epoch.
    """
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches
    
    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch
    
    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)
