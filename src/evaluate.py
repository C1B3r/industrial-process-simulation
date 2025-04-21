import sys
import os

# Add 'models' folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

# Importing the Generator and Discriminator classes from the models folder
from generator import Generator
from discriminator import Discriminator

import torch
import matplotlib.pyplot as plt
import pandas as pd  # Add pandas to handle CSV files

# Load trained models
generator = Generator()
discriminator = Discriminator()

# Assuming the models have been saved in the 'models/saved' directory
generator.load_state_dict(torch.load('models/saved/generator_epoch100.pt'))
discriminator.load_state_dict(torch.load('models/saved/discriminator_epoch100.pt'))

# Generate synthetic data using the trained generator
latent_dim = 16
z = torch.randn(100, latent_dim)  # Generate 100 synthetic samples
synthetic_data = generator(z)

# Visualize the generated data
plt.figure(figsize=(12, 6))

# Plot synthetic data distribution
plt.subplot(1, 2, 1)
plt.hist(synthetic_data.detach().numpy().flatten(), bins=50, alpha=0.6, color='g')
plt.title('Generated Data Distribution')

# Attempt to load real data (test_data.pt) correctly
real_data = None

try:
    # Try loading the file as a PyTorch tensor
    real_data = torch.load('data/processed/test_data.pt', weights_only=False)
    print(f"Real data loaded as PyTorch tensor: {real_data.shape}")
except Exception as e:
    print(f"Error loading real data as PyTorch tensor: {e}")
    
    try:
        # If the PyTorch tensor loading fails, try loading it as a CSV file
        real_data = pd.read_csv('data/processed/test_data.pt')  # Assuming this is actually a CSV file
        print(f"Real data loaded as CSV with shape: {real_data.shape}")
    except Exception as e:
        print(f"Error loading real data as CSV: {e}")

# Plot real data distribution (if real_data is successfully loaded)
if isinstance(real_data, torch.Tensor):
    plt.subplot(1, 2, 2)
    plt.hist(real_data.numpy().flatten(), bins=50, alpha=0.6, color='r')  # Ensure it's a numpy array
    plt.title('Real Data Distribution')
elif isinstance(real_data, pd.DataFrame):
    plt.subplot(1, 2, 2)
    plt.hist(real_data.to_numpy().flatten(), bins=50, alpha=0.6, color='r')
    plt.title('Real Data Distribution    (CSV)')

plt.show()
