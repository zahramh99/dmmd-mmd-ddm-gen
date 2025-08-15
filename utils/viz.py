import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def save_images(tensor, filename, nrow=8, padding=2):
    """Save a grid of images to file"""
    grid = make_grid(tensor, nrow=nrow, padding=padding, normalize=True)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_loss_curve(losses, filename="loss_curve.png"):
    """Plot training loss curve"""
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.savefig(filename)
    plt.close()

def feature_visualization(features, labels, filename="features.png"):
    """2D projection of feature space (using PCA/t-SNE)"""
    from sklearn.manifold import TSNE
    
    # Reduce to 2D
    tsne = TSNE(n_components=2)
    features_2d = tsne.fit_transform(features.cpu().numpy())
    
    # Plot
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels.cpu().numpy())
    plt.colorbar()
    plt.savefig(filename)
    plt.close()