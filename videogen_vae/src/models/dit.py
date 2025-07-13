"""
Diffusion Transformer (DiT) for Video Generation
Based on latest research from ViD-GPT, FullDiT, and other state-of-the-art models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple, List
import math


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for timesteps"""
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


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization with timestep and optional text conditioning"""
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.linear = nn.Linear(cond_dim, dim * 2)
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B*T, num_patches, dim], cond: [B, cond_dim]
        B_cond = cond.shape[0]
        B_x = x.shape[0]

        # Expand condition to match x's batch dimension
        if B_x != B_cond:
            # Repeat condition for each time step (B*T / B = T)
            cond = cond.unsqueeze(1).expand(-1, B_x // B_cond, -1).reshape(B_x, -1)

        scale, shift = self.linear(cond).chunk(2, dim=-1)
        # scale, shift: [B*T, dim] -> [B*T, 1, dim] for broadcasting with x: [B*T, num_patches, dim]
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        return self.norm(x) * (1 + scale) + shift


class FlashAttention(nn.Module):
    """Multi-head attention with Flash Attention support"""
    def __init__(self, dim: int, num_heads: int = 8, head_dim: int = 64, dropout: float = 0.):
        super().__init__()
        inner_dim = head_dim * num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)

        # Use Flash Attention if available, otherwise standard attention
        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class DiTBlock(nn.Module):
    """Diffusion Transformer block with adaptive normalization"""
    def __init__(self, dim: int, num_heads: int, head_dim: int, cond_dim: int, dropout: float = 0.):
        super().__init__()
        self.norm1 = AdaptiveLayerNorm(dim, cond_dim)
        self.attn = FlashAttention(dim, num_heads, head_dim, dropout)
        self.norm2 = AdaptiveLayerNorm(dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x, cond), mask)
        x = x + self.mlp(self.norm2(x, cond))
        return x


class SpatioTemporalAttention(nn.Module):
    """Spatio-temporal attention for video generation"""
    def __init__(self, dim: int, num_heads: int, head_dim: int, dropout: float = 0.):
        super().__init__()
        self.spatial_attn = FlashAttention(dim, num_heads, head_dim, dropout)
        self.temporal_attn = FlashAttention(dim, num_heads, head_dim, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, B: int, T: int, H: int, W: int) -> torch.Tensor:
        # Spatial attention (within each frame)
        x_spatial = rearrange(x, '(b t) (h w) c -> (b t) (h w) c', b=B, t=T, h=H, w=W)
        x_spatial = self.spatial_attn(x_spatial)

        # Temporal attention (across frames)
        x_temporal = rearrange(x_spatial, '(b t) (h w) c -> (b h w) t c', b=B, t=T, h=H, w=W)
        x_temporal = self.temporal_attn(x_temporal)
        x_out = rearrange(x_temporal, '(b h w) t c -> (b t) (h w) c', b=B, h=H, w=W)

        return self.norm(x_out + x)


class VideoDiT(nn.Module):
    """Video Diffusion Transformer"""
    def __init__(
        self,
        video_size: Tuple[int, int, int],  # (frames, height, width)
        patch_size: int = 4,
        in_channels: int = 3,
        hidden_size: int = 512,
        depth: int = 24,
        num_heads: int = 8,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        learn_sigma: bool = True,
        use_text_cond: bool = True,
        text_dim: int = 768,
    ):
        super().__init__()
        self.video_size = video_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.learn_sigma = learn_sigma
        self.use_text_cond = use_text_cond

        T, H, W = video_size
        assert H % patch_size == 0 and W % patch_size == 0
        self.num_patches = (T * H * W) // (patch_size ** 2)
        self.patch_dim = in_channels * patch_size ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size
        )

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        self.time_embed = SinusoidalPosEmb(hidden_size)

        # Conditioning
        cond_dim = hidden_size
        if use_text_cond:
            self.text_proj = nn.Linear(text_dim, hidden_size)
            cond_dim += hidden_size
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size * 4)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, head_dim, hidden_size * 4, dropout)
            for _ in range(depth)
        ])

        # Spatio-temporal attention layers (every 3rd block)
        self.st_blocks = nn.ModuleList([
            SpatioTemporalAttention(hidden_size, num_heads, head_dim, dropout)
            if i % 3 == 2 else nn.Identity()
            for i in range(depth)
        ])

        # Output projection
        self.norm_out = nn.LayerNorm(hidden_size)
        out_channels = in_channels * (2 if learn_sigma else 1)
        self.out_proj = nn.Linear(hidden_size, self.patch_dim)

        self._init_weights()

    def _init_weights(self):
        # Initialize weights
        nn.init.normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def unpatchify(self, x: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        """Convert patches back to video"""
        # x shape: [B*T, num_patches, patch_dim]
        B = x.shape[0] // T
        p = self.patch_size
        h, w = H // p, W // p
        c = 3  # Always 3 channels for RGB

        # Verify dimensions
        expected_patch_dim = self.patch_dim
        if x.shape[-1] != expected_patch_dim:
            raise ValueError(f"Patch dimension mismatch: expected {expected_patch_dim}, got {x.shape[-1]}")

        # Reshape to [B, T, h, w, patch_dim]
        x = x.reshape(B, T, h, w, -1)

        # Reshape patch_dim to [p, p, c]
        x = x.reshape(B, T, h, w, p, p, c)

        # Rearrange to [B, T, c, H, W]
        x = torch.einsum('bthwpqc->btchpwq', x)
        x = x.reshape(B, T, c, H, W)
        return x

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, C, H, W = x.shape

        # Patchify and embed
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.patch_embed(x)
        x = rearrange(x, '(b t) c h w -> (b t) (h w) c', b=B, t=T)

        # Add positional embeddings
        x = x + self.pos_embed[:, :x.shape[1]].to(x.dtype)

        # Prepare conditioning
        t_emb = self.time_embed(t).to(x.dtype)
        if self.use_text_cond and text_emb is not None:
            text_proj = self.text_proj(text_emb).to(x.dtype)
            t_emb = torch.cat([t_emb, text_proj], dim=-1)
        cond = self.cond_proj(t_emb)

        # Apply transformer blocks
        for block, st_block in zip(self.blocks, self.st_blocks):
            x = block(x, cond, mask)
            if not isinstance(st_block, nn.Identity):
                x = st_block(x, B, T, H // self.patch_size, W // self.patch_size)

        # Output projection
        x = self.norm_out(x)
        x = self.out_proj(x)

        # Unpatchify
        x = self.unpatchify(x, T, H, W)

        return x


def create_dit_model(config: dict) -> VideoDiT:
    """Create DiT model from config"""
    video_size = (
        config['model']['num_frames'],
        config['model']['frame_size'],
        config['model']['frame_size']
    )

    return VideoDiT(
        video_size=video_size,
        patch_size=config['model'].get('patch_size', 16),
        hidden_size=config['model']['dim'],
        depth=config['model']['depth'],
        num_heads=config['model']['num_heads'],
        head_dim=config['model']['head_dim'],
        mlp_ratio=config['model'].get('mlp_ratio', 4.0),
        dropout=config['model'].get('dropout', 0.0),
        learn_sigma=config['model'].get('learn_sigma', False),
        use_text_cond=config['model']['use_text_conditioning'],
    ) 