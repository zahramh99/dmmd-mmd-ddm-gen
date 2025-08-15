import torch
import torch.nn as nn
import torch.nn.functional as F

class DCGenerator(nn.Module):
    """DCGAN-style generator for 32x32 or 64x64 images."""
    def __init__(self, z_dim=128, img_size=32, channels=3):
        super().__init__()
        self.z_dim = z_dim
        self.img_size = img_size
        self.channels = channels
        
        # Determine number of upsampling blocks needed
        if img_size == 32:
            self.n_blocks = 3
        elif img_size == 64:
            self.n_blocks = 4
        else:
            raise ValueError(f"Unsupported image size: {img_size}")
            
        # Initial dense layer
        self.fc = nn.Linear(z_dim, 512 * (img_size // (2**self.n_blocks))**2)
        
        # Convolutional blocks
        self.conv_blocks = nn.ModuleList()
        in_channels = 512
        for i in range(self.n_blocks):
            out_channels = in_channels // 2
            self.conv_blocks.append(nn.Sequential)(
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )
            in_channels = out_channels
            
        # Final layer
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, channels, 4, 2, 1),
            nn.Tanh()
        )
        
    def forward(self, z):
        # Project and reshape
        x = self.fc(z)
        x = x.view(-1, 512, self.img_size // (2**self.n_blocks), 
                          self.img_size // (2**self.n_blocks))
        
        # Apply conv blocks
        for block in self.conv_blocks:
            x = block(x)
            
        # Final layer
        return self.final_conv(x)