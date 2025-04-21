import pytest
import torch
from models.generator import Generator
from models.discriminator import Discriminator

def test_generator_forward():
    generator = Generator()
    z = torch.randn(64, 16)  # Latent space size is 16
    generated_data = generator(z)
    assert generated_data.shape == (64, 10)  # Output should have 10 features (sensor data)
    
def test_discriminator_forward():
    discriminator = Discriminator()
    data = torch.randn(64, 10)  # Fake data with 10 features (sensor data)
    output = discriminator(data)
    assert output.shape == (64, 1)  # Discriminator outputs a probability for each sample
    assert output.min() >= 0 and output.max() <= 1  # Sigmoid output should be between 0 and 1
