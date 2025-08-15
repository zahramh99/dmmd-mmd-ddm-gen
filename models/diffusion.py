import torch
import torch.nn as nn
import numpy as np
from typing import Union, Tuple

class GaussianDiffusion:
    """Gaussian diffusion process utilities."""
    def __init__(self, timesteps: int = 1000, beta_schedule: str = "linear"):
        self.timesteps = timesteps
        
        if beta_schedule == "linear":
            self.betas = torch.linspace(1e-4, 0.02, timesteps)
        elif beta_schedule == "cosine":
            s = 0.008
            steps = torch.arange(timesteps + 1)
            x = (steps / timesteps + s) / (1 + s) * np.pi / 2
            alphas_cumprod = torch.cos(x) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            self.betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
            
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion process: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x0)
            
        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        
        return sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise, noise
    
    def _extract(self, vals: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int]) -> torch.Tensor:
        """Extract values for specific timesteps."""
        batch_size = t.shape[0]
        out = vals.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

class UNet(nn.Module):
    """Simple UNet for diffusion models."""
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()
        
        # Down blocks
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.MaxPool2d(2)
        )
        
        self.down2 = self._make_down_block(base_channels, 2*base_channels)
        self.down3 = self._make_down_block(2*base_channels, 4*base_channels)
        
        # Middle
        self.mid = nn.Sequential(
            nn.Conv2d(4*base_channels, 4*base_channels, 3, padding=1),
            nn.GroupNorm(8, 4*base_channels),
            nn.SiLU(),
            nn.Conv2d(4*base_channels, 4*base_channels, 3, padding=1),
            nn.GroupNorm(8, 4*base_channels),
            nn.SiLU()
        )
        
        # Up blocks
        self.up1 = self._make_up_block(4*base_channels, 2*base_channels)
        self.up2 = self._make_up_block(2*base_channels, base_channels)
        
        self.final = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )
        
    def _make_down_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.MaxPool2d(2)
        )
    
    def _make_up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )
    
    def forward(self, x, t=None):
        # Downsample
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        
        # Middle
        x_mid = self.mid(x3)
        
        # Upsample
        x = self.up1(x_mid + x3)
        x = self.up2(x + x2)
        
        return self.final(x + x1)