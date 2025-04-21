import pytest
import torch
from train import train_loop
from models.generator import Generator
from models.discriminator import Discriminator

def test_training_loss():
    generator = Generator()
    discriminator = Discriminator()
    
    z = torch.randn(64, 16)  # Latent space size
    real_data = torch.randn(64, 10)  # Real sensor data with 10 features
    fake_data = generator(z)  # Generate fake data
    
    # Ensure the losses are computed correctly (just checking that they are finite numbers)
    loss_D = train_loop.compute_discriminator_loss(discriminator, real_data, fake_data)
    loss_G = train_loop.compute_generator_loss(discriminator, fake_data)
    
    assert loss_D.isfinite().all()
    assert loss_G.isfinite().all()

def test_optimizer_update():
    generator = Generator()
    discriminator = Discriminator()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

    # Check if the optimizer updates the model parameters
    old_generator_weights = generator.state_dict()
    old_discriminator_weights = discriminator.state_dict()

    z = torch.randn(64, 16)
    fake_data = generator(z)
    loss_G = train_loop.compute_generator_loss(discriminator, fake_data)

    optimizer_G.zero_grad()
    loss_G.backward()
    optimizer_G.step()

    assert not torch.equal(old_generator_weights['model.0.weight'], generator.state_dict()['model.0.weight'])
    assert not torch.equal(old_discriminator_weights['model.0.weight'], discriminator.state_dict()['model.0.weight'])
