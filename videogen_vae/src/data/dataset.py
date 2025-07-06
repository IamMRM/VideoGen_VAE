"""
Modern Video Dataset with efficient loading, caching, and augmentation
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Any
import cv2
import random
from pathlib import Path
import hashlib
from concurrent.futures import ThreadPoolExecutor
import logging
from torchvision import transforms
import albumentations as A


logger = logging.getLogger(__name__)


class VideoDataset(Dataset):
    """Modern video dataset with caching and augmentation"""
    
    def __init__(
        self,
        data_dir: str,
        resolution: int = 512,
        fps: int = 24,
        duration: float = 20.0,
        cache_dir: Optional[str] = None,
        cache_latents: bool = True,
        augment: bool = True,
        num_workers: int = 4,
        video_extensions: List[str] = ['.mp4', '.avi', '.mov', '.webm'],
        caption_extension: str = '.txt',
        use_bucketing: bool = True,
        bucket_sizes: Optional[List[Tuple[int, int]]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.fps = fps
        self.duration = duration
        self.num_frames = int(fps * duration)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_latents = cache_latents
        self.augment = augment
        self.num_workers = num_workers
        self.video_extensions = video_extensions
        self.caption_extension = caption_extension
        self.use_bucketing = use_bucketing
        
        # Setup buckets for different aspect ratios
        if bucket_sizes is None:
            self.bucket_sizes = [
                (512, 512),   # 1:1
                (576, 448),   # 9:7
                (608, 416),   # 19:13
                (640, 384),   # 5:3
                (704, 320),   # 11:5
                (768, 256),   # 3:1
            ]
        else:
            self.bucket_sizes = bucket_sizes
            
        # Find all video files
        self.video_files = self._find_video_files()
        logger.info(f"Found {len(self.video_files)} video files")
        
        # Setup cache
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
        # Setup augmentation pipeline
        self.transform = self._setup_augmentation()
        
    def _find_video_files(self) -> List[Path]:
        """Find all video files in the data directory"""
        video_files = []
        for ext in self.video_extensions:
            video_files.extend(self.data_dir.rglob(f"*{ext}"))
        return sorted(video_files)
        
    def _setup_augmentation(self) -> A.Compose:
        """Setup video augmentation pipeline"""
        if not self.augment:
            return A.Compose([
                A.Resize(self.resolution, self.resolution),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            
        return A.Compose([
            A.OneOf([
                A.RandomResizedCrop(self.resolution, self.resolution, scale=(0.8, 1.0)),
                A.Resize(self.resolution, self.resolution),
            ], p=0.5),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
            ], p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.1),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
    def _get_cache_path(self, video_path: Path, prefix: str = "") -> Path:
        """Get cache file path for a video"""
        if not self.cache_dir:
            return None
            
        # Create unique hash for video file
        video_hash = hashlib.md5(str(video_path).encode()).hexdigest()
        cache_name = f"{prefix}_{video_hash}_{self.resolution}_{self.fps}_{self.num_frames}.pt"
        return self.cache_dir / cache_name
        
    def _load_video(self, video_path: Path) -> np.ndarray:
        """Load video frames"""
        cap = cv2.VideoCapture(str(video_path))
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame indices to sample
        if total_frames < self.num_frames:
            # Loop video if too short
            frame_indices = list(range(total_frames)) * (self.num_frames // total_frames + 1)
            frame_indices = frame_indices[:self.num_frames]
        else:
            # Sample frames uniformly
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # Use last valid frame if read fails
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8))
                    
        cap.release()
        
        # Ensure we have the right number of frames
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8))
            
        return np.array(frames[:self.num_frames])
        
    def _load_caption(self, video_path: Path) -> str:
        """Load caption for video"""
        caption_path = video_path.with_suffix(self.caption_extension)
        
        if caption_path.exists():
            with open(caption_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        else:
            # Use filename as caption if no caption file
            return video_path.stem.replace('_', ' ')
            
    def _apply_augmentation(self, frames: np.ndarray) -> torch.Tensor:
        """Apply augmentation to video frames"""
        augmented_frames = []
        
        for frame in frames:
            augmented = self.transform(image=frame)['image']
            augmented_frames.append(augmented)
            
        # Convert to tensor [T, C, H, W]
        frames_tensor = torch.from_numpy(np.array(augmented_frames)).float()
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)
        
        return frames_tensor
        
    def _get_bucket_size(self, width: int, height: int) -> Tuple[int, int]:
        """Get appropriate bucket size for given dimensions"""
        if not self.use_bucketing:
            return (self.resolution, self.resolution)
            
        aspect_ratio = width / height
        
        # Find closest bucket
        best_bucket = self.bucket_sizes[0]
        min_diff = float('inf')
        
        for bucket_w, bucket_h in self.bucket_sizes:
            bucket_ratio = bucket_w / bucket_h
            diff = abs(aspect_ratio - bucket_ratio)
            
            if diff < min_diff:
                min_diff = diff
                best_bucket = (bucket_w, bucket_h)
                
        return best_bucket
        
    def __len__(self) -> int:
        return len(self.video_files)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_path = self.video_files[idx]
        
        # Check cache first
        cache_path = self._get_cache_path(video_path, "processed")
        
        if cache_path and cache_path.exists():
            try:
                data = torch.load(cache_path, map_location='cpu')
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_path}: {e}")
                
        # Load video
        frames = self._load_video(video_path)
        
        # Get original dimensions for bucketing
        if self.use_bucketing and len(frames) > 0:
            h, w = frames[0].shape[:2]
            target_size = self._get_bucket_size(w, h)
            
            # Update transform with new size
            self.transform[0].height = target_size[1]
            self.transform[0].width = target_size[0]
        else:
            target_size = (self.resolution, self.resolution)
            
        # Apply augmentation
        frames_tensor = self._apply_augmentation(frames)
        
        # Load caption
        caption = self._load_caption(video_path)
        
        # Prepare output
        data = {
            'pixel_values': frames_tensor,
            'text': caption,
            'video_path': str(video_path),
            'resolution': target_size,
        }
        
        # Save to cache
        if cache_path:
            try:
                torch.save(data, cache_path)
            except Exception as e:
                logger.warning(f"Failed to save cache {cache_path}: {e}")
                
        return data


class VideoCollator:
    """Custom collator for video batches with bucketing support"""
    
    def __init__(self, tokenizer=None, max_length: int = 77):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Group by resolution for bucketing
        resolution_groups = {}
        for item in batch:
            res = item['resolution']
            if res not in resolution_groups:
                resolution_groups[res] = []
            resolution_groups[res].append(item)
            
        # Process each resolution group
        all_pixel_values = []
        all_texts = []
        all_paths = []
        
        for resolution, items in resolution_groups.items():
            pixel_values = torch.stack([item['pixel_values'] for item in items])
            texts = [item['text'] for item in items]
            paths = [item['video_path'] for item in items]
            
            all_pixel_values.append(pixel_values)
            all_texts.extend(texts)
            all_paths.extend(paths)
            
        # Concatenate all pixel values
        pixel_values = torch.cat(all_pixel_values, dim=0)
        
        # Tokenize texts if tokenizer provided
        if self.tokenizer:
            text_inputs = self.tokenizer(
                all_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
        else:
            text_inputs = all_texts
            
        return {
            'pixel_values': pixel_values,
            'text_inputs': text_inputs,
            'paths': all_paths,
        }


def create_dataloader(config: dict, tokenizer=None):
    """Create dataloader from config"""
    dataset = VideoDataset(
        data_dir=config['data']['dataset_path'],
        resolution=config['data']['resolution'],
        fps=config['data']['fps'],
        duration=config['data']['duration'],
        cache_dir=config['data']['cache_dir'] if config['data']['cache_latents'] else None,
        augment=config['data'].get('augment', True),
        num_workers=config['data']['num_workers'],
    )
    
    collator = VideoCollator(tokenizer=tokenizer)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )
    
    return dataloader 