import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import os
import yaml

from models.diffusion import GaussianDiffusion, UNet
from mmd.losses import mmd2_unbiased
from mmd.kernels import MMDKernel
from utils.datasets import get_cifar10
from utils.seed import set_seed

class MMDDDMTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Set seed
        set_seed(config['seed'])
        
        # Diffusion process
        self.diffusion = GaussianDiffusion(
            timesteps=config['timesteps'],
            beta_schedule=config['beta_schedule']
        )
        
        # Load pretrained UNet
        self.model = UNet(
            in_channels=3,
            out_channels=3,
            base_channels=config['base_channels']
        ).to(self.device)
        
        # Load pretrained weights if provided
        if config['pretrained_path']:
            state_dict = torch.load(config['pretrained_path'], map_location=self.device)
            self.model.load_state_dict(state_dict)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['lr'],
            betas=(config['beta1'], config['beta2'])
        )
        
        # Kernel for MMD
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
            
            # Sample timesteps uniformly
            t = torch.randint(
                0, self.config['inference_steps'],
                (x_real.size(0),),
                device=self.device
            )
            
            # Generate samples with current model
            x_gen = self._ddim_sample(x_real.size(0), steps=self.config['inference_steps'])
            
            # Compute MMD loss
            loss = mmd2_unbiased(x_real.view(x_real.size(0), -1),
                                x_gen.view(x_gen.size(0), -1),
                                self.kernel)
            
            # Update model
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
    
    def _ddim_sample(self, n_samples: int, steps: int = 10) -> torch.Tensor:
        """DDIM sampling with current model."""
        x = torch.randn(n_samples, 3, self.config['img_size'], self.config['img_size'], 
                        device=self.device)
        
        timesteps = torch.linspace(
            self.diffusion.timesteps - 1, 0, steps + 1
        ).long().to(self.device)
        
        for i in range(steps):
            t = timesteps[i]
            next_t = timesteps[i+1] if i < steps - 1 else -1
            
            # Predict noise
            with torch.no_grad():
                pred_noise = self.model(x, t)
                
            # Compute x0 estimate
            alpha_cumprod_t = self.diffusion._extract(
                self.diffusion.alphas_cumprod, t, x.shape
            )
            sqrt_one_minus_alpha_cumprod_t = self.diffusion._extract(
                self.diffusion.sqrt_one_minus_alphas_cumprod, t, x.shape
            )
            x0 = (x - sqrt_one_minus_alpha_cumprod_t * pred_noise) / torch.sqrt(alpha_cumprod_t)
            
            if next_t == -1:
                x = x0
            else:
                alpha_cumprod_next_t = self.diffusion._extract(
                    self.diffusion.alphas_cumprod, next_t, x.shape
                )
                x = torch.sqrt(alpha_cumprod_next_t) * x0 + \
                    torch.sqrt(1 - alpha_cumprod_next_t) * pred_noise
                    
        return x
    
    def _save_samples(self, step):
        with torch.no_grad():
            samples = self._ddim_sample(64, steps=self.config['inference_steps'])
            save_image(
                samples,
                os.path.join(self.config['output_dir'], f"samples_{step}.png"),
                nrow=8,
                normalize=True
            )
    
    def _save_checkpoint(self, step):
        torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, os.path.join(self.config['output_dir'], f"ckpt_{step}.pt"))

if __name__ == "__main__":
    # Load config
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)['mmd_ddm']
    
    # Train
    trainer = MMDDDMTrainer(config)
    trainer.train()