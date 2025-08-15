import torch
import yaml
from tqdm import tqdm
from models import UNet, GaussianDiffusion
from utils import get_cifar10_loader, save_images

class DiffusionTrainer:
    def __init__(self, config_path="configs/default.yaml"):
        self.config = self._load_config(config_path)
        self.device = torch.device(self.config['device'])
        
        self.model = UNet(base_channels=self.config['base_channels']).to(self.device)
        self.diffusion = GaussianDiffusion(timesteps=self.config['timesteps'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
        self.train_loader = get_cifar10_loader(self.config['batch_size'])
        
    def train(self, num_epochs):
        for epoch in tqdm(range(num_epochs)):
            for x0, _ in self.train_loader:
                x0 = x0.to(self.device)
                
                # Sample random timesteps
                t = torch.randint(0, self.diffusion.timesteps, (x0.size(0),), device=self.device)
                
                # Forward process
                xt, noise = self.diffusion.q_sample(x0, t)
                
                # Predict noise
                pred_noise = self.model(xt, t)
                
                # Compute loss
                loss = torch.mean((pred_noise - noise)**2)
                
                # Update model
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            if epoch % self.config['save_every'] == 0:
                self._save_samples(epoch)

    def _save_samples(self, epoch, num_samples=16):
        samples = self.diffusion.p_sample_loop(
            self.model, 
            shape=(num_samples, 3, 32, 32)
        )
        save_images(samples, f"diffusion_samples_{epoch}.png")
        
def main():
    trainer = DiffusionTrainer()
    trainer.train(num_epochs=100)

if __name__ == "__main__":
    main()