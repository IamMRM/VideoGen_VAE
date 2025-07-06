"""
Modern inference script for video generation
Supports batch generation, multiple sampling methods, and various output formats
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
import argparse
import logging
from tqdm import tqdm
import cv2
import imageio
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dit import create_dit_model
from models.diffusion import VideoDiffusion


logger = logging.getLogger(__name__)


class VideoGenerator:
    """Video generation inference class"""
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.dtype = dtype
        
        # Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load config
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = checkpoint.get('config', {})
            
        # Initialize model
        self.setup_model(checkpoint)
        
    def setup_model(self, checkpoint: Dict[str, Any]):
        """Setup model from checkpoint"""
        # Create model
        self.model = create_dit_model(self.config)
        
        # Load weights
        if 'ema_state_dict' in checkpoint:
            logger.info("Loading EMA model weights")
            self.model.load_state_dict(checkpoint['ema_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        # Create diffusion wrapper
        self.diffusion = VideoDiffusion(
            model=self.model,
            num_timesteps=self.config['model']['timesteps'],
            loss_type=self.config['model']['loss_type'],
            beta_schedule=self.config['model']['beta_schedule'],
            use_ema=False,  # Already loaded EMA weights
        )
        
        # Move to device and set dtype
        self.diffusion = self.diffusion.to(self.device)
        if self.dtype == torch.float16:
            self.diffusion = self.diffusion.half()
            
        # Set eval mode
        self.diffusion.eval()
        
        # Setup text encoder if needed
        if self.config['model']['use_text_conditioning']:
            self.setup_text_encoder()
            
    def setup_text_encoder(self):
        """Setup text encoder for conditioning"""
        logger.info("Setting up text encoder")
        
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        
        self.text_encoder = self.text_encoder.to(self.device)
        if self.dtype == torch.float16:
            self.text_encoder = self.text_encoder.half()
            
        self.text_encoder.eval()
        
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        """Encode text prompts"""
        if not hasattr(self, 'text_encoder'):
            return None
            
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(**inputs).last_hidden_state
            
        return text_embeddings
        
    @torch.no_grad()
    def generate(
        self,
        prompts: Optional[List[str]] = None,
        num_videos: int = 1,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        use_ddim: bool = True,
        eta: float = 0.0,
        seed: Optional[int] = None,
        progress: bool = True,
    ) -> torch.Tensor:
        """Generate videos"""
        
        # Set dimensions
        if num_frames is None:
            num_frames = self.config['model']['num_frames']
        if height is None:
            height = self.config['model']['frame_size']
        if width is None:
            width = self.config['model']['frame_size']
            
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        # Prepare prompts
        if prompts is None:
            prompts = [""] * num_videos
        elif len(prompts) == 1:
            prompts = prompts * num_videos
        elif len(prompts) != num_videos:
            raise ValueError(f"Number of prompts ({len(prompts)}) must match num_videos ({num_videos})")
            
        # Encode text
        text_embeddings = self.encode_text(prompts) if prompts[0] else None
        
        # Classifier-free guidance
        if guidance_scale > 1.0 and text_embeddings is not None:
            # Create unconditional embeddings
            uncond_embeddings = self.encode_text([""] * num_videos)
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            
        # Generate shape
        shape = (
            num_videos,
            num_frames,
            self.config['model']['channels'],
            height,
            width
        )
        
        # Generate
        if use_ddim:
            samples = self.diffusion.ddim_sample(
                shape,
                text_embeddings,
                ddim_timesteps=num_inference_steps,
                eta=eta,
                device=self.device,
                progress=progress
            )
        else:
            samples = self.diffusion.sample(
                shape,
                text_embeddings,
                device=self.device,
                progress=progress
            )
            
        # Apply guidance
        if guidance_scale > 1.0 and text_embeddings is not None:
            # Split conditional and unconditional
            samples_uncond, samples_cond = samples.chunk(2)
            samples = samples_uncond + guidance_scale * (samples_cond - samples_uncond)
            
        # Convert to [0, 1] range
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        return samples
        
    def save_videos(
        self,
        videos: torch.Tensor,
        output_dir: str,
        prefix: str = "video",
        fps: int = 24,
        format: str = "mp4",
        codec: str = "h264",
    ):
        """Save generated videos to disk"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to numpy
        videos = videos.cpu().numpy()
        
        for i, video in enumerate(videos):
            # Rearrange dimensions: [T, C, H, W] -> [T, H, W, C]
            video = np.transpose(video, (0, 2, 3, 1))
            
            # Convert to uint8
            video = (video * 255).astype(np.uint8)
            
            # Save video
            output_path = output_dir / f"{prefix}_{i:04d}.{format}"
            
            if format == "mp4":
                self._save_mp4(video, output_path, fps, codec)
            elif format == "gif":
                self._save_gif(video, output_path, fps)
            elif format == "webm":
                self._save_webm(video, output_path, fps)
            else:
                raise ValueError(f"Unknown format: {format}")
                
            logger.info(f"Saved video: {output_path}")
            
    def _save_mp4(self, frames: np.ndarray, output_path: Path, fps: int, codec: str):
        """Save video as MP4"""
        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec=codec,
            quality=9,
            pixelformat='yuv420p'
        )
        
        for frame in frames:
            writer.append_data(frame)
            
        writer.close()
        
    def _save_gif(self, frames: np.ndarray, output_path: Path, fps: int):
        """Save video as GIF"""
        # Convert to PIL images
        pil_frames = [Image.fromarray(frame) for frame in frames]
        
        # Save as GIF
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / fps),
            loop=0
        )
        
    def _save_webm(self, frames: np.ndarray, output_path: Path, fps: int):
        """Save video as WebM"""
        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec='libvpx-vp9',
            quality=9,
        )
        
        for frame in frames:
            writer.append_data(frame)
            
        writer.close()
        
    def save_frames(
        self,
        videos: torch.Tensor,
        output_dir: str,
        prefix: str = "frame",
    ):
        """Save individual frames as images"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to numpy
        videos = videos.cpu().numpy()
        
        for video_idx, video in enumerate(videos):
            video_dir = output_dir / f"video_{video_idx:04d}"
            video_dir.mkdir(exist_ok=True)
            
            # Rearrange dimensions: [T, C, H, W] -> [T, H, W, C]
            video = np.transpose(video, (0, 2, 3, 1))
            
            # Convert to uint8
            video = (video * 255).astype(np.uint8)
            
            # Save frames
            for frame_idx, frame in enumerate(video):
                frame_path = video_dir / f"{prefix}_{frame_idx:04d}.png"
                Image.fromarray(frame).save(frame_path)
                
            logger.info(f"Saved frames to: {video_dir}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Generate videos using trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, help="Path to config file (optional)")
    parser.add_argument("--prompts", type=str, nargs="+", help="Text prompts for generation")
    parser.add_argument("--num-videos", type=int, default=1, help="Number of videos to generate")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--num-frames", type=int, help="Number of frames per video")
    parser.add_argument("--height", type=int, help="Video height")
    parser.add_argument("--width", type=int, help="Video width")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--format", type=str, default="mp4", choices=["mp4", "gif", "webm"], help="Output format")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")
    parser.add_argument("--save-frames", action="store_true", help="Save individual frames")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create generator
    generator = VideoGenerator(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        dtype=torch.float16 if args.fp16 else torch.float32,
    )
    
    # Generate videos
    logger.info("Generating videos...")
    videos = generator.generate(
        prompts=args.prompts,
        num_videos=args.num_videos,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    
    # Save videos
    logger.info("Saving videos...")
    generator.save_videos(
        videos,
        output_dir=args.output_dir,
        fps=args.fps,
        format=args.format,
    )
    
    # Save frames if requested
    if args.save_frames:
        logger.info("Saving frames...")
        generator.save_frames(videos, output_dir=args.output_dir)
        
    logger.info("Done!")
    

if __name__ == "__main__":
    main() 