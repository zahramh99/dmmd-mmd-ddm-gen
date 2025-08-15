# Generator models
from .generator import (
    DCGenerator,        # DCGAN-style generator for DMMD
    ResNetGenerator      # Optional ResNet-based generator
)

# Diffusion models
from .diffusion import (
    GaussianDiffusion,   # Diffusion process utilities
    UNet                 # U-Net architecture for diffusion
)

# Optional: Version info
__version__ = "0.1.0"
__description__ = "PyTorch models for MMD-based generative modeling"

# Explicitly define what gets imported with `from models import *`
__all__ = [
    'DCGenerator',
    'ResNetGenerator',
    'GaussianDiffusion',
    'UNet'
]
