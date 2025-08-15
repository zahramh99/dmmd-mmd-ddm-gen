import torch
import torch.nn as nn
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance

class Evaluator:
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        
        # KID metric
        self.kid = KernelInceptionDistance(subset_size=50).to(self.device)
        
        # FID metric
        self.fid = FrechetInceptionDistance().to(self.device)
        
    def compute_kid(self, real_images, gen_images):
        """Compute Kernel Inception Distance."""
        self.kid.update(real_images, real=True)
        self.kid.update(gen_images, real=False)
        return self.kid.compute()
        
    def compute_fid(self, real_images, gen_images):
        """Compute Frechet Inception Distance."""
        self.fid.update(real_images, real=True)
        self.fid.update(gen_images, real=False)
        return self.fid.compute()
        
    def compute_mmd(self, real_features, gen_features, kernel_fn):
        """Compute MMD between feature representations."""
        return mmd2_unbiased(real_features, gen_features, kernel_fn)