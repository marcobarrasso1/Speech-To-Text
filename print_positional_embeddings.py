import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math


def sinusoids1(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0, "Channels must be even for sinusoidal positional encoding."
    
    # Compute the log timescale increment
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    
    # Compute inverse timescales
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    
    # Generate scaled time positions
    scaled_time = torch.arange(length, dtype=torch.float32).unsqueeze(1) * inv_timescales.unsqueeze(0)
    
    # Compute sin and cos and concatenate
    pos_emb = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    
    return pos_emb  # Shape: (length, channels)
def visualize_positional_embeddings(length=50, channels=128):
    """Visualize sinusoidal positional embeddings"""
    embeddings = sinusoids(length, channels)
    
    plt.figure(figsize=(15, 5))
    
    # Heatmap of embeddings
    plt.subplot(121)
    sns.heatmap(embeddings.numpy(), cmap='viridis', center=0, 
                xticklabels=False, yticklabels=False)
    plt.title('Positional Embeddings Heatmap')
    
    # Line plot of first few dimensions
    plt.subplot(122)
    for i in range(10):  # Plot first 10 dimensions
        plt.plot(embeddings[:, i].numpy(), label=f'Dim {i}')
    plt.title('First 10 Embedding Dimensions')
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.legend(loc='best', ncol=2)
    
    plt.tight_layout()
    plt.show()

# Example usage
visualize_positional_embeddings()
