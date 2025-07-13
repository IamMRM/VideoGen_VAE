"""
Simplified Diffusion Transformer for Video Generation
Optimized for single-activity datasets with reduced memory footprint
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SimpleAttention(nn.Module):
    """Memory-efficient simplified attention mechanism"""
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Memory optimization: process in chunks if sequence is too long
        if N > 1024:  # If more than 1024 tokens, use chunked attention
            return self._chunked_attention(x)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Use memory-efficient attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

    def _chunked_attention(self, x: torch.Tensor, chunk_size: int = 512) -> torch.Tensor:
        """Process attention in chunks to save memory"""
        B, N, C = x.shape

        # Process QKV for entire sequence
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Process attention in chunks
        outputs = []
        for i in range(0, N, chunk_size):
            end_idx = min(i + chunk_size, N)
            q_chunk = q[:, :, i:end_idx]

            # Compute attention for this chunk against all keys
            attn_chunk = (q_chunk @ k.transpose(-2, -1)) * self.scale
            attn_chunk = attn_chunk.softmax(dim=-1)
            attn_chunk = self.dropout(attn_chunk)

            # Apply attention to values
            out_chunk = (attn_chunk @ v).transpose(1, 2).reshape(B, end_idx - i, C)
            outputs.append(out_chunk)

        x = torch.cat(outputs, dim=1)
        x = self.proj(x)
        return x


class SimpleMLP(nn.Module):
    """Simple MLP block"""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleTransformerBlock(nn.Module):
    """Simplified transformer block with gradient checkpointing support"""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.0, use_checkpoint: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SimpleAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SimpleMLP(dim, int(dim * mlp_ratio), dropout)
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleVideoDiT(nn.Module):
    """Simplified Video Diffusion Transformer"""
    def __init__(
        self,
        video_size: Tuple[int, int, int],  # (frames, height, width)
        patch_size: int = 8,
        in_channels: int = 3,
        hidden_size: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        learn_sigma: bool = False,  # Disabled for simplicity
    ):
        super().__init__()
        self.video_size = video_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.learn_sigma = learn_sigma

        T, H, W = video_size
        assert H % patch_size == 0 and W % patch_size == 0

        # Calculate number of patches
        self.num_patches = (T * H * W) // (patch_size ** 2)
        self.patch_dim = in_channels * patch_size ** 2

        # Patch embedding - simplified
        self.patch_embed = nn.Linear(self.patch_dim, hidden_size)

        # Positional embeddings - make it larger to handle variable sizes
        max_patches = self.num_patches * 2  # Allow for some flexibility
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, hidden_size))
        self.time_embed = SinusoidalPosEmb(hidden_size)

        # Time conditioning
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        # Transformer blocks with gradient checkpointing
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(hidden_size, num_heads, mlp_ratio, dropout, use_checkpoint=True)
            for _ in range(depth)
        ])

        # Output layers
        self.norm_out = nn.LayerNorm(hidden_size)
        out_channels = in_channels * (2 if learn_sigma else 1)
        self.out_proj = nn.Linear(hidden_size, out_channels * patch_size ** 2)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.pos_embed, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert video to patches"""
        B, T, C, H, W = x.shape
        p = self.patch_size

        # Reshape to patches
        x = x.reshape(B, T, C, H // p, p, W // p, p)
        x = x.permute(0, 1, 3, 5, 2, 4, 6)  # [B, T, H//p, W//p, C, p, p]
        x = x.reshape(B, T * (H // p) * (W // p), C * p * p)

        # Ensure dtype is preserved
        return x

    def unpatchify(self, x: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        """Convert patches back to video"""
        B = x.shape[0]
        p = self.patch_size
        h, w = H // p, W // p
        c = 3 if not self.learn_sigma else 6

        # Calculate expected output channels from input
        patch_elements = x.shape[-1]  # Last dimension after linear layer
        expected_patch_elements = c * p * p

        if patch_elements != expected_patch_elements:
            # Adjust channels if mismatch
            c = patch_elements // (p * p)

        # Reshape patches back to video
        x = x.reshape(B, T, h, w, c, p, p)
        x = x.permute(0, 1, 4, 2, 5, 3, 6)  # [B, T, C, h, p, w, p]
        x = x.reshape(B, T, c, H, W)

        return x

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,  # Ignored for simplicity
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, C, H, W = x.shape

        # Convert to patches
        x = self.patchify(x)  # [B, N, patch_dim]

        # Embed patches
        x = self.patch_embed(x)

        # Add positional embeddings with safety check
        if x.shape[1] > self.pos_embed.shape[1]:
            # If we need more positional embeddings, interpolate
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=x.shape[1],
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            # Ensure interpolated tensor matches input dtype
            pos_embed = pos_embed.to(x.dtype)
        else:
            pos_embed = self.pos_embed[:, :x.shape[1]]

        x = x + pos_embed
        
        # Time conditioning
        t_emb = self.time_embed(t)
        # Ensure time embeddings match model dtype
        t_emb = t_emb.to(x.dtype)
        t_emb = self.time_mlp(t_emb)
        
        # Add time conditioning to all tokens
        x = x + t_emb.unsqueeze(1)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Output projection
        x = self.norm_out(x)
        x = self.out_proj(x)
        
        # Convert back to video
        x = self.unpatchify(x, T, H, W)
        
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=2)
            
        return x


def create_simple_dit_model(config: dict) -> SimpleVideoDiT:
    """Create simplified DiT model from config"""
    video_size = (
        config['model']['num_frames'],
        config['model']['frame_size'],
        config['model']['frame_size']
    )
    
    return SimpleVideoDiT(
        video_size=video_size,
        patch_size=config['model'].get('patch_size', 8),
        hidden_size=config['model']['dim'],
        depth=config['model']['depth'],
        num_heads=config['model']['num_heads'],
        mlp_ratio=config['model'].get('mlp_ratio', 2.0),
        dropout=config['model'].get('dropout', 0.0),
        learn_sigma=config['model'].get('learn_sigma', False),
    ) 