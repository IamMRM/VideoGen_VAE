"""
Exponential Moving Average (EMA) for model parameters
"""

import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional


class EMAModel:
    """
    Exponential Moving Average of model parameters
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[str] = None,
        use_ema_warmup: bool = True,
        inv_gamma: float = 1.0,
        power: float = 2/3,
    ):
        """
        Args:
            model: Model to track
            decay: EMA decay rate
            device: Device for EMA model
            use_ema_warmup: Use EMA warmup
            inv_gamma: Inverse gamma for EMA warmup
            power: Power for EMA warmup
        """
        self.decay = decay
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.optimization_step = 0
        
        # Create EMA model
        self.ema_model = deepcopy(model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()
        
        if device:
            self.ema_model = self.ema_model.to(device)
            
    def get_decay(self) -> float:
        """
        Get current decay rate with optional warmup
        """
        if not self.use_ema_warmup:
            return self.decay
            
        # EMA warmup formula
        step = self.optimization_step
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        
        return min(self.decay, value)
        
    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update EMA parameters
        """
        self.optimization_step += 1
        decay = self.get_decay()
        
        # Update parameters
        for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
            
        # Update buffers
        for ema_buffer, buffer in zip(self.ema_model.buffers(), model.buffers()):
            ema_buffer.data.copy_(buffer.data)
            
    def set(self, model: nn.Module):
        """
        Set EMA parameters to model
        """
        self.ema_model.load_state_dict(model.state_dict())
        
    def state_dict(self):
        """
        Get EMA model state dict
        """
        return self.ema_model.state_dict()
        
    def load_state_dict(self, state_dict):
        """
        Load EMA model state dict
        """
        self.ema_model.load_state_dict(state_dict) 