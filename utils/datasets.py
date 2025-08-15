import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

def get_cifar10(batch_size=64, img_size=32, num_workers=4, root="data/"):
    """Get CIFAR-10 dataloaders."""
    tfm = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train = torchvision.datasets.CIFAR10(
        root, train=True, transform=tfm, download=True
    )
    test = torchvision.datasets.CIFAR10(
        root, train=False, transform=tfm, download=True
    )
    
    return (
        DataLoader(train, batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(test, batch_size, shuffle=False, num_workers=num_workers)
    )

def get_celeba(batch_size=64, img_size=64, num_workers=4, root="data/"):
    """Get CelebA dataloaders."""
    tfm = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = torchvision.datasets.CelebA(
        root, split='all', transform=tfm, download=True
    )
    
    # Split into train/test
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train, test = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    return (
        DataLoader(train, batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(test, batch_size, shuffle=False, num_workers=num_workers)
    )