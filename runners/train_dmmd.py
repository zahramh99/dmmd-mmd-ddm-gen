import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import os
import yaml

from models.generator import DCGenerator
from mmd.losses import mmd2_unbiased
from mmd.kernels import MMDKernel
from utils.datasets import get_cifar10
from utils.seed import set_seed

class DMMDTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Set seed
        set_seed(config['seed'])
        
        # Create models
        self.generator = DCGenerator(
            z_dim=config['z_dim'],
            img_size=config['img_size'],
            channels=config['channels']
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config['lr'],
            betas=(config['beta1'], config['beta2'])
        )
        
        # Kernel
        self.kernel = MMDKernel(
            kernel_type='rbf',
            sigmas=config['sigmas']
        )
        
        # Data
        self.train_loader, _ = get_cifar10(
            batch_size=config['batch_size'],
            img_size=config['img_size']
        )
        
        # Create output dir
        os.makedirs(config['output_dir'], exist_ok=True)
        
    def train(self):
        pbar = tqdm(range(self.config['n_iters']))
        
        for step in pbar:
            # Sample real data
            x_real = next(iter(self.train_loader))[0].to(self.device)
            
            # Sample noise and generate
            z = torch.randn(x_real.size(0), self.config['z_dim'], device=self.device)
            x_gen = self.generator(z)
            
            # Compute MMD loss
            loss = mmd2_unbiased(x_real.view(x_real.size(0), -1),
                                x_gen.view(x_gen.size(0), -1),
                                self.kernel)
            
            # Update generator
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Logging
            if step % self.config['log_every'] == 0:
                pbar.set_description(f"Step {step}: MMD^2 {loss.item():.4f}")
                
            # Save samples
            if step % self.config['sample_every'] == 0:
                self._save_samples(step)
                
            # Save checkpoint
            if step % self.config['ckpt_every'] == 0:
                self._save_checkpoint(step)
    
    def _save_samples(self, step):
        with torch.no_grad():
            z = torch.randn(64, self.config['z_dim'], device=self.device)
            samples = self.generator(z)
            save_image(
                samples,
                os.path.join(self.config['output_dir'], f"samples_{step}.png"),
                nrow=8,
                normalize=True
            )
    
    def _save_checkpoint(self, step):
        torch.save({
            'step': step,
            'generator_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, os.path.join(self.config['output_dir'], f"ckpt_{step}.pt"))

if __name__ == "__main__":
    # Load config
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)['dmmd']
    
    # Train
    trainer = DMMDTrainer(config)
    trainer.train()