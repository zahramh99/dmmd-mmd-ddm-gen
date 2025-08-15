import torch

def mmd2_unbiased(x: torch.Tensor, y: torch.Tensor, kernel_fn) -> torch.Tensor:
    """Compute unbiased squared MMD between x and y.
    
    Args:
        x: (n_samples_x, n_features)
        y: (n_samples_y, n_features)
        kernel_fn: Function that computes kernel matrix
        
    Returns:
        mmd2: Unbiased estimate of squared MMD
    """
    n, m = x.size(0), y.size(0)
    
    # Compute kernel matrices
    k_xx = kernel_fn(x, x)
    k_yy = kernel_fn(y, y)
    k_xy = kernel_fn(x, y)
    
    # Remove diagonals for unbiased estimate
    k_xx = k_xx - torch.diag(torch.diag(k_xx))
    k_yy = k_yy - torch.diag(torch.diag(k_yy))
    
    # Compute MMD^2
    term_xx = k_xx.sum() / (n * (n - 1))
    term_yy = k_yy.sum() / (m * (m - 1))
    term_xy = 2 * k_xy.sum() / (n * m)
    
    return term_xx + term_yy - term_xy

def mmd2_biased(x: torch.Tensor, y: torch.Tensor, kernel_fn) -> torch.Tensor:
    """Compute biased squared MMD between x and y."""
    n, m = x.size(0), y.size(0)
    
    k_xx = kernel_fn(x, x).mean()
    k_yy = kernel_fn(y, y).mean()
    k_xy = kernel_fn(x, y).mean()
    
    return k_xx + k_yy - 2 * k_xy