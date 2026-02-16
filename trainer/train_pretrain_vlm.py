import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from model.model_vlm import Qwen3VLM, VLMConfig
from dataset.lm_dataset import VLMDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, init_distributed_mode, setup_seed, init_vlm_model, vlm_checkpoint, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    Training loop for a single epoch.
    
    Args:
        epoch: Current epoch number.
        loader: DataLoader.
        iters: Total iterations per epoch.
        start_step: Step to resume from (if resuming within an epoch).
        wandb: WandB logger instance.
    """
    start_time = time.time()
    for step, (input_ids, labels, pixel_values) in enumerate(loader, start=start_step + 1):
        # Move data to GPU
        # input_ids: [batch_size, seq_len]
        # pixel_values: [batch_size, channels, height, width] (or 5D with num_images)
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        pixel_values = pixel_values.to(args.device)
        
        # Update Learning Rate
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward Pass with Mixed Precision
        with autocast_ctx:
            res = model(input_ids, labels=labels, pixel_values=pixel_values)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        # Backward Pass
        scaler.scale(loss).backward()

        # Optimizer Step (Gradient Accumulation)
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        # Logging
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # Checkpointing
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if vlm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{vlm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # Filter out frozen vision encoder weights
            clean_state_dict = {
                key: value for key, value in state_dict.items() if not key.startswith('vision_encoder.')
            }
            clean_state_dict = {k: v.half().cpu() for k, v in clean_state_dict.items()}  # Save as FP16 on CPU
            torch.save(clean_state_dict, ckp)
            vlm_checkpoint(vlm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='./checkpoints', scaler=scaler)
            model.train()
            del state_dict, clean_state_dict

        del input_ids, labels, pixel_values, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-V Pretrain")
    parser.add_argument("--save_dir", type=str, default="./out", help="Directory to save model weights")
    parser.add_argument('--save_weight', default='pretrain_vlm', type=str, help="Prefix for saved weight files")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=4e-4, help="Initial learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Precision (bfloat16 or float16)")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval (steps)")
    parser.add_argument("--save_interval", type=int, default=1000, help="Checkpoint saving interval (steps)")
    parser.add_argument('--hidden_size', default=512, type=int, help="Model hidden size")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="Number of hidden layers")
    parser.add_argument('--max_seq_len', default=640, type=int, help="Max sequence length for training")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="Enable Mixture of Experts (0=No, 1=Yes)")
    parser.add_argument('--use_deepstack', default=0, type=int, choices=[0, 1], help="Enable DeepStack (0=No, 1=Yes)")
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_i2t.parquet", help="Path to training data (Parquet)")
    parser.add_argument('--from_weight', default='llm', type=str, help="Base weights to initialize from (or 'none')")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="Resume from checkpoint (0=No, 1=Yes)")
    parser.add_argument('--freeze_llm', default=1, type=int, choices=[0, 1], help="Freeze LLM backbone (Stage 1 specific)")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="Use torch.compile (0=No, 1=Yes)")
    parser.add_argument("--use_wandb", default=1, action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="Qwen3-V-Pretrain", help="WandB project name")
    parser.add_argument('--model_path', type=str, default='../Models/Qwen3-0.6B', help="Path to Qwen3 model (tokenizer & base)")
    parser.add_argument('--enable_thinking', default=1, type=int, choices=[0, 1], help="Enable/Disable thinking mode (0=No, 1=Yes)")
    args = parser.parse_args()

    # ========== 1. Environment & Seed Setup ==========
    local_rank = init_distributed_mode()
    Logger(f"Process launched. Rank: {local_rank}")
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. Configuration & Checkpoint ==========
    os.makedirs(args.save_dir, exist_ok=True)
    vlm_config = VLMConfig(use_moe=args.use_moe, use_deepstack=args.use_deepstack, max_seq_len=args.max_seq_len, enable_thinking=args.enable_thinking)
    Logger(f"VLM Config created. MOE: {args.use_moe}, DeepStack: {args.use_deepstack}, Thinking: {args.enable_thinking}")
    # Disable thinking if requested
    if args.enable_thinking == 0:
        if hasattr(vlm_config, 'enable_thinking'):
             vlm_config.enable_thinking = False
        if hasattr(vlm_config, 'generation_config') and hasattr(vlm_config.generation_config, 'enable_thinking'):
             vlm_config.generation_config.enable_thinking = False
    # Load resume data if enabled
    ckp_data = vlm_checkpoint(vlm_config, weight=args.save_weight, save_dir='./checkpoints') if args.from_resume==1 else None
    
    # ========== 3. Mixed Precision Setup ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. WandB Setup ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"Qwen3-V-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. Model, Tokenizer, Data, Optimizer ==========
    # Initialize Model
    Logger("Initializing model...")
    model, tokenizer, preprocess = init_vlm_model(vlm_config, from_weight=args.from_weight, tokenizer_path=args.model_path, device=args.device, freeze_llm=bool(args.freeze_llm))
    Logger("Model initialized.")
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    
    # Initialize Dataset
    Logger(f"Loading dataset from {args.data_path}...")
    train_ds = VLMDataset(args.data_path, tokenizer, preprocess=preprocess,
                          image_special_token=vlm_config.image_special_token,
                          max_length=vlm_config.max_seq_len)
    Logger(f"Dataset loaded. Size: {len(train_ds)}")
    
    # Initialize DDP Sampler
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    # Initialize Scaler for Mixed Precision
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    
    # Initialize Optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    
    # ========== 6. Resume from Checkpoint ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. Wrap Model with DDP ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. Start Training ==========
    Logger(f"Starting training loop for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        
        # Skip steps if resuming
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: Skipping {start_step} steps, starting from step {start_step + 1}')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. Cleanup ==========
    if dist.is_initialized(): dist.destroy_process_group()
