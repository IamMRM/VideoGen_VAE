"""
Modern training script for video generation models
Supports multi-GPU training, mixed precision, and advanced monitoring
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import wandb
from pathlib import Path
from typing import Dict, Optional, Any
import logging
from tqdm import tqdm
import time
from datetime import datetime
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import CLIPTextModel, CLIPTokenizer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dit import create_dit_model
from models.simple_dit import create_simple_dit_model
from models.diffusion import VideoDiffusion
from data.dataset import create_dataloader
from utils.ema import EMAModel
from utils.metrics import calculate_fvd, calculate_is


logger = logging.getLogger(__name__)


class VideoGenerationTrainer:
    """Modern trainer for video generation models"""

    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup accelerator for multi-GPU training
        self.accelerator = Accelerator(
            mixed_precision=self.config['training']['mixed_precision'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            log_with=["tensorboard", "wandb"] if self.config['training']['use_wandb'] else ["tensorboard"],
            project_dir=self.config['project']['output_dir'],
        )

        # Set seed for reproducibility
        set_seed(self.config['project']['seed'])

        # Setup directories
        self.setup_directories()

        # Setup logging
        self.setup_logging()

        # Initialize model and training components
        self.setup_model()
        self.setup_data()
        self.setup_optimization()
        self.setup_monitoring()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_fvd = float('inf')

    def setup_directories(self):
        """Create necessary directories"""
        self.output_dir = Path(self.config['project']['output_dir'])
        self.checkpoint_dir = Path(self.config['project']['checkpoint_dir'])
        self.log_dir = Path(self.config['project']['log_dir'])

        for dir_path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def setup_model(self):
        """Initialize model and diffusion wrapper"""
        logger.info("Setting up model...")

        # Create base model (use simplified version for better memory efficiency)
        architecture = self.config['model'].get('architecture', 'dit')
        if architecture == 'simple_dit':
            self.model = create_simple_dit_model(self.config)
        else:
            self.model = create_dit_model(self.config)

        # Create diffusion wrapper
        self.diffusion = VideoDiffusion(
            model=self.model,
            num_timesteps=self.config['model']['timesteps'],
            loss_type=self.config['model']['loss_type'],
            beta_schedule=self.config['model']['beta_schedule'],
            use_ema=True,
        )

        # Setup text encoder if using text conditioning
        if self.config['model']['use_text_conditioning']:
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

            # Freeze text encoder
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # Move to device
        self.diffusion = self.accelerator.prepare(self.diffusion)
        if hasattr(self, 'text_encoder'):
            self.text_encoder = self.accelerator.prepare(self.text_encoder)

        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

    def setup_data(self):
        """Setup data loaders"""
        logger.info("Setting up data loaders...")

        # Create dataloader
        self.train_dataloader = create_dataloader(
            self.config,
            tokenizer=self.tokenizer if hasattr(self, 'tokenizer') else None
        )

        # Prepare with accelerator
        self.train_dataloader = self.accelerator.prepare(self.train_dataloader)

        # Calculate steps
        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config['training']['num_epochs']

        logger.info(f"Train dataset size: {len(self.train_dataloader.dataset)}")
        logger.info(f"Steps per epoch: {self.steps_per_epoch}")
        logger.info(f"Total training steps: {self.total_steps}")

    def setup_optimization(self):
        """Setup optimizer and scheduler"""
        logger.info("Setting up optimization...")

        # Create optimizer
        optimizer_config = self.config['training']
        if optimizer_config['optimizer'] == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=float(optimizer_config['learning_rate']),
                weight_decay=float(optimizer_config['weight_decay']),
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config['optimizer']}")

        # Create scheduler
        if optimizer_config['scheduler'] == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.total_steps,
                eta_min=float(optimizer_config['learning_rate']) * 0.1
            )
        elif optimizer_config['scheduler'] == 'constant':
            from torch.optim.lr_scheduler import ConstantLR
            self.scheduler = ConstantLR(self.optimizer, factor=1.0)
        else:
            raise ValueError(f"Unknown scheduler: {optimizer_config['scheduler']}")

        # Prepare with accelerator
        self.optimizer, self.scheduler = self.accelerator.prepare(self.optimizer, self.scheduler)

    def setup_monitoring(self):
        """Setup monitoring and logging"""
        if self.accelerator.is_main_process:
            # Initialize trackers
            run_name = f"{self.config['project']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Filter config for TensorBoard compatibility (only basic types)
            def filter_config(obj):
                if isinstance(obj, dict):
                    return {k: filter_config(v) for k, v in obj.items()
                           if isinstance(v, (int, float, str, bool))}
                elif isinstance(obj, (int, float, str, bool)):
                    return obj
                else:
                    return str(obj)  # Convert complex types to string

            filtered_config = filter_config(self.config)

            self.accelerator.init_trackers(
                project_name=self.config['training']['wandb_project'],
                config=filtered_config,
                init_kwargs={"wandb": {"name": run_name}}
            )

    def encode_text(self, texts: list) -> torch.Tensor:
        """Encode text prompts"""
        if not hasattr(self, 'text_encoder'):
            return None

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(self.accelerator.device)

        with torch.no_grad():
            text_embeddings = self.text_encoder(**inputs).last_hidden_state

        return text_embeddings

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single training step"""
        # Get data
        videos = batch['pixel_values']

        # Encode text if available
        text_embeddings = None
        if 'text_inputs' in batch and hasattr(self, 'text_encoder'):
            if isinstance(batch['text_inputs'], dict):
                with torch.no_grad():
                    text_embeddings = self.text_encoder(**batch['text_inputs']).last_hidden_state
            else:
                text_embeddings = self.encode_text(batch['text_inputs'])

        # Forward pass
        with self.accelerator.autocast():
            loss = self.diffusion(videos, text_embeddings)

        # Backward pass
        self.accelerator.backward(loss)

        # Gradient clipping
        gradient_clip_val = float(self.config['training']['gradient_clip_val'])
        if gradient_clip_val > 0:
            self.accelerator.clip_grad_norm_(
                self.model.parameters(),
                gradient_clip_val
            )

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        # Update EMA
        diffusion_model = self.accelerator.unwrap_model(self.diffusion)
        if diffusion_model.use_ema:
            diffusion_model.update_ema()

        # Metrics
        metrics = {
            'loss': loss.item(),
            'lr': self.scheduler.get_last_lr()[0],
        }

        return metrics

    def validate(self) -> Dict[str, float]:
        """Validation step"""
        logger.info("Running validation...")

        # Get a batch of real videos for comparison
        real_videos = None
        for batch in self.train_dataloader:
            real_videos = batch['pixel_values']
            break

        # Generate sample videos
        sample_shape = (
            4,  # batch size
            self.config['model']['num_frames'],
            self.config['model']['channels'],
            self.config['model']['frame_size'],
            self.config['model']['frame_size']
        )

        # Generate videos
        with torch.no_grad():
            diffusion_model = self.accelerator.unwrap_model(self.diffusion)
            if self.config['inference']['num_inference_steps'] < 100:
                # Use DDIM for faster sampling
                samples = diffusion_model.ddim_sample(
                    sample_shape,
                    None,  # No text conditioning
                    ddim_timesteps=self.config['inference']['num_inference_steps'],
                    device=self.accelerator.device,
                    progress=False
                )
            else:
                # Use standard DDPM sampling
                samples = diffusion_model.sample(
                    sample_shape,
                    None,  # No text conditioning
                    device=self.accelerator.device,
                    progress=False
                )

        # Calculate actual metrics
        metrics = {}

        if real_videos is not None:
            # Calculate FVD
            fvd_score = calculate_fvd(real_videos, samples, device=self.accelerator.device)
            metrics['fvd'] = fvd_score

            # Calculate IS
            is_mean, is_std = calculate_is(samples, device=self.accelerator.device)
            metrics['is_score'] = is_mean
            metrics['is_std'] = is_std

            # Calculate validation loss
            val_loss = self.calculate_validation_loss(real_videos, samples)
            metrics['val_loss'] = val_loss
        else:
            metrics = {
                'val_loss': 0.0,
                'fvd': 0.0,
                'is_score': 0.0,
            }

        # Log sample videos
        if self.accelerator.is_main_process:
            # Convert to valid range [0, 1]
            samples_normalized = (samples + 1) / 2
            samples_normalized = torch.clamp(samples_normalized, 0, 1)

            # Log to wandb/tensorboard (only if wandb is enabled)
            if self.config['training']['use_wandb']:
                self.accelerator.log({
                    "samples": wandb.Video(samples_normalized.cpu().numpy(), fps=self.config['inference']['fps'])
                }, step=self.global_step)

        return metrics

    def calculate_validation_loss(self, real_videos: torch.Tensor, generated_videos: torch.Tensor) -> float:
        """Calculate validation loss between real and generated videos"""
        # Use MSE loss between real and generated videos
        loss = torch.nn.functional.mse_loss(real_videos, generated_videos)
        return loss.item()

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        if not self.accelerator.is_main_process:
            return

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
        }

        # Save EMA model if available
        diffusion_model = self.accelerator.unwrap_model(self.diffusion)
        if diffusion_model.use_ema:
            checkpoint['ema_state_dict'] = diffusion_model.ema_model.state_dict()
            
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.epoch:04d}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            
        # Keep only recent checkpoints
        self.cleanup_checkpoints()
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
    def cleanup_checkpoints(self, keep_last: int = 5):
        """Remove old checkpoints"""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        
        if len(checkpoints) > keep_last:
            for checkpoint in checkpoints[:-keep_last]:
                checkpoint.unlink()
                
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Training loop
        progress_bar = tqdm(
            total=self.total_steps,
            disable=not self.accelerator.is_local_main_process,
            desc="Training"
        )
        
        for epoch in range(self.config['training']['num_epochs']):
            self.epoch = epoch
            epoch_metrics = []
            
            # Training
            self.model.train()
            for batch_idx, batch in enumerate(self.train_dataloader):
                # Train step
                metrics = self.train_step(batch)
                epoch_metrics.append(metrics)
                
                # Update progress
                self.global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(metrics)
                
                # Log metrics
                if self.global_step % self.config['training']['log_every_n_steps'] == 0:
                    avg_metrics = {
                        k: np.mean([m[k] for m in epoch_metrics[-100:]])
                        for k in metrics.keys()
                    }
                    self.accelerator.log(avg_metrics, step=self.global_step)
                    
            # Validation
            if (epoch + 1) % self.config['training']['validate_every_n_epochs'] == 0:
                val_metrics = self.validate()
                self.accelerator.log(val_metrics, step=self.global_step)
                
                # Check if best model
                is_best = val_metrics.get('fvd', float('inf')) < self.best_fvd
                if is_best:
                    self.best_fvd = val_metrics['fvd']
                    
            # Save checkpoint
            if (epoch + 1) % self.config['training']['save_every_n_epochs'] == 0:
                self.save_checkpoint(is_best=False)
                
        # Final checkpoint
        self.save_checkpoint(is_best=False)
        
        # End training
        self.accelerator.end_training()
        logger.info("Training completed!")
        
        
def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train video generation model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = VideoGenerationTrainer(args.config)
    trainer.train()
    

if __name__ == "__main__":
    main() 