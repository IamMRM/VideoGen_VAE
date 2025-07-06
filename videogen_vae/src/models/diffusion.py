"""
Video Diffusion Model with modern techniques
Incorporates Flow Matching, DDIM, and other advanced sampling methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
from tqdm import tqdm
import math


class NoiseSchedule:
    """Modern noise scheduling with various schedules"""
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule_type: str = "cosine"
    ):
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        
        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "cosine":
            s = 0.008
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        elif schedule_type == "sigmoid":
            betas = torch.sigmoid(torch.linspace(-6, 6, num_timesteps))
            betas = (betas - betas.min()) / (betas.max() - betas.min())
            betas = betas * (beta_end - beta_start) + beta_start
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
            
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
        
    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Register buffer for device handling"""
        setattr(self, name, tensor)
        
    def to(self, device):
        """Move all buffers to device"""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor):
                setattr(self, attr_name, attr.to(device))
        return self


class VideoDiffusion(nn.Module):
    """Video Diffusion Model with modern techniques"""
    
    def __init__(
        self,
        model: nn.Module,
        num_timesteps: int = 1000,
        loss_type: str = "l2",
        beta_schedule: str = "cosine",
        parameterization: str = "eps",  # "eps" or "v"
        use_ema: bool = True,
        ema_decay: float = 0.9999,
    ):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps
        self.loss_type = loss_type
        self.parameterization = parameterization
        self.use_ema = use_ema
        
        # Initialize noise schedule
        self.noise_schedule = NoiseSchedule(num_timesteps, schedule_type=beta_schedule)
        
        # EMA model
        if use_ema:
            self.ema_model = self._create_ema_model()
            self.ema_decay = ema_decay
        else:
            self.ema_model = None
            
    def _create_ema_model(self):
        """Create exponential moving average model"""
        import copy
        ema_model = copy.deepcopy(self.model)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model
        
    def update_ema(self):
        """Update EMA model parameters"""
        if self.ema_model is None:
            return
            
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
                
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self._extract(self.noise_schedule.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.noise_schedule.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size):
        """Extract values from a 1D tensor"""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
        
    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """Predict x_0 from x_t and noise"""
        sqrt_recip_alphas_cumprod_t = self._extract(self.noise_schedule.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = self._extract(self.noise_schedule.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise
        
    def predict_noise_from_start(self, x_t: torch.Tensor, t: torch.Tensor, x_0: torch.Tensor):
        """Predict noise from x_t and x_0"""
        sqrt_recip_alphas_cumprod_t = self._extract(self.noise_schedule.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = self._extract(self.noise_schedule.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return (sqrt_recip_alphas_cumprod_t * x_t - x_0) / sqrt_recipm1_alphas_cumprod_t
        
    def p_losses(
        self, 
        x_start: torch.Tensor, 
        t: torch.Tensor, 
        text_emb: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None
    ):
        """Calculate training loss"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # Model prediction
        model_output = self.model(x_noisy, t, text_emb)
        
        # Calculate target based on parameterization
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            # v-parameterization from Imagen Video
            v = self.noise_schedule.sqrt_alphas_cumprod[t] * noise - self.noise_schedule.sqrt_one_minus_alphas_cumprod[t] * x_start
            target = v
        else:
            raise ValueError(f"Unknown parameterization: {self.parameterization}")
            
        # Calculate loss
        if self.loss_type == "l1":
            loss = F.l1_loss(model_output, target, reduction="none")
        elif self.loss_type == "l2":
            loss = F.mse_loss(model_output, target, reduction="none")
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(model_output, target, reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
            
        return loss.mean()
        
    def forward(self, x: torch.Tensor, text_emb: Optional[torch.Tensor] = None):
        """Training forward pass"""
        b = x.shape[0]
        device = x.device
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        
        return self.p_losses(x, t, text_emb)
        
    @torch.no_grad()
    def p_sample(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        text_emb: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
        return_pred_x0: bool = False
    ):
        """Single denoising step"""
        model = self.ema_model if self.use_ema and not self.training else self.model
        
        # Model prediction
        model_output = model(x, t, text_emb)
        
        # Get x_0 prediction
        if self.parameterization == "eps":
            pred_x0 = self.predict_start_from_noise(x, t, model_output)
        elif self.parameterization == "v":
            # Convert v to eps first
            v = model_output
            eps = self.noise_schedule.sqrt_one_minus_alphas_cumprod[t] * v + self.noise_schedule.sqrt_alphas_cumprod[t] * x
            pred_x0 = self.predict_start_from_noise(x, t, eps)
        
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
        # Get posterior distribution
        posterior_mean = (
            self._extract(self.noise_schedule.posterior_mean_coef1, t, x.shape) * pred_x0 +
            self._extract(self.noise_schedule.posterior_mean_coef2, t, x.shape) * x
        )
        posterior_variance = self._extract(self.noise_schedule.posterior_variance, t, x.shape)
        posterior_log_variance = self._extract(self.noise_schedule.posterior_log_variance_clipped, t, x.shape)
        
        # Sample
        noise = torch.randn_like(x) if t[0] > 0 else 0
        pred = posterior_mean + torch.exp(0.5 * posterior_log_variance) * noise
        
        if return_pred_x0:
            return pred, pred_x0
        return pred
        
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        text_emb: Optional[torch.Tensor] = None,
        device: str = "cuda",
        progress: bool = True,
        return_intermediates: bool = False
    ):
        """Generate samples using DDPM sampling"""
        x = torch.randn(shape, device=device)
        intermediates = []
        
        timesteps = list(reversed(range(0, self.num_timesteps)))
        if progress:
            timesteps = tqdm(timesteps, desc="Sampling")
            
        for t in timesteps:
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch, text_emb)
            
            if return_intermediates and t % 100 == 0:
                intermediates.append(x.clone())
                
        if return_intermediates:
            return x, intermediates
        return x
        
    @torch.no_grad()
    def ddim_sample(
        self,
        shape: Tuple[int, ...],
        text_emb: Optional[torch.Tensor] = None,
        ddim_timesteps: int = 50,
        eta: float = 0.0,
        device: str = "cuda",
        progress: bool = True
    ):
        """DDIM sampling for faster generation"""
        # Select subset of timesteps
        c = self.num_timesteps // ddim_timesteps
        ddim_timestep_seq = np.asarray(list(range(0, self.num_timesteps, c)))
        
        x = torch.randn(shape, device=device)
        
        timesteps = reversed(ddim_timestep_seq)
        if progress:
            timesteps = tqdm(timesteps, desc="DDIM Sampling")
            
        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Model prediction
            model = self.ema_model if self.use_ema else self.model
            model_output = model(x, t_batch, text_emb)
            
            # Get x_0 prediction
            if self.parameterization == "eps":
                pred_x0 = self.predict_start_from_noise(x, t_batch, model_output)
                eps = model_output
            elif self.parameterization == "v":
                v = model_output
                eps = self.noise_schedule.sqrt_one_minus_alphas_cumprod[t] * v + self.noise_schedule.sqrt_alphas_cumprod[t] * x
                pred_x0 = self.predict_start_from_noise(x, t_batch, eps)
                
            # DDIM update
            if i < len(ddim_timestep_seq) - 1:
                t_next = ddim_timestep_seq[i + 1]
                alpha_bar = self.noise_schedule.alphas_cumprod[t]
                alpha_bar_next = self.noise_schedule.alphas_cumprod[t_next]
                
                sigma = eta * torch.sqrt((1 - alpha_bar_next) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_next)
                
                mean_pred = (
                    torch.sqrt(alpha_bar_next) * pred_x0 +
                    torch.sqrt(1 - alpha_bar_next - sigma ** 2) * eps
                )
                
                noise = torch.randn_like(x) if eta > 0 else 0
                x = mean_pred + sigma * noise
            else:
                x = pred_x0
                
        return x 