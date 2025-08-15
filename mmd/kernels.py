import torch
import numpy as np
from typing import List, Union

class RBFKernel:
    def __init__(self, sigmas: Union[List[float], torch.Tensor]):
        """RBF kernel with multiple bandwidths.
        
        Args:
            sigmas: List or tensor of bandwidth parameters
        """
        if isinstance(sigmas, list):
            sigmas = torch.tensor(sigmas)
        self.sigmas = sigmas.reshape(-1, 1, 1)  # (n_sigmas, 1, 1)
        
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel matrix between x and y.
        
        Args:
            x: (n_samples_x, n_features)
            y: (n_samples_y, n_features)
            
        Returns:
            kernel_matrix: (n_samples_x, n_samples_y)
        """
        if x.dim() != 2 or y.dim() != 2:
            raise ValueError("Inputs must be 2D tensors")
            
        # Compute squared distances
        x_norm = (x**2).sum(1).view(-1, 1)
        y_norm = (y**2).sum(1).view(1, -1)
        dists = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
        
        # Compute RBF kernel for each bandwidth
        gamma = 1.0 / (2.0 * self.sigmas**2)
        k_vals = torch.exp(-gamma * dists)
        
        # Average over all bandwidths
        return k_vals.mean(0)

class MMDKernel:
    def __init__(self, kernel_type='rbf', **kwargs):
        """Wrapper for different kernel types."""
        if kernel_type == 'rbf':
            self.kernel = RBFKernel(**kwargs)
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")
            
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.kernel(x, y)