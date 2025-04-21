import pytest
import torch
from evaluate import generate_synthetic_data
from models.generator import Generator

def test_generate_synthetic_data():
    generator = Generator()
    generator.load_state_dict(torch.load('models/saved/generator_epoch100.pt'))
    
    # Generate synthetic data
    z = torch.randn(100, 16)  # Latent space size
    synthetic_data = generate_synthetic_data(generator, z)
    
    assert synthetic_data.shape == (100, 10)  # Ensure it has the correct number of features
    assert synthetic_data.min() >= -1 and synthetic_data.max() <= 1  # Check if data is normalized
