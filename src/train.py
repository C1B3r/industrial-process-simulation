import torch
import torch.optim as optim
from models.generator import Generator
from models.discriminator import Discriminator
from data_loader import create_dataloader
import matplotlib.pyplot as plt

# Hyperparameters
latent_dim = 16
epochs = 100
batch_size = 64
lr = 0.0002

# Initialize models
generator = Generator()
discriminator = Discriminator()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# Loss function
criterion = torch.nn.BCELoss()

# Create DataLoader (assuming processed data is already available)
train_data = load_data('data/processed/train_data.pt')  # Example path
train_dataloader = create_dataloader(train_data, batch_size)

# Training loop
for epoch in range(epochs):
    for real_data in train_dataloader:
        # Train discriminator
        optimizer_D.zero_grad()
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        real_data = real_data[0]

        # Forward pass real data through discriminator
        output_real = discriminator(real_data)
        loss_real = criterion(output_real, real_labels)

        # Generate fake data
        z = torch.randn(batch_size, latent_dim)
        fake_data = generator(z)

        # Forward pass fake data through discriminator
        output_fake = discriminator(fake_data.detach())
        loss_fake = criterion(output_fake, fake_labels)

        # Backpropagate discriminator loss
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        # Train generator
        optimizer_G.zero_grad()
        output_fake = discriminator(fake_data)
        loss_G = criterion(output_fake, real_labels)  # Try to fool the discriminator
        loss_G.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch}/{epochs}], D Loss: {loss_D.item():.4f}, G Loss: {loss_G.item():.4f}")
    
    # Save models periodically
    if epoch % 10 == 0:
        torch.save(generator.state_dict(), f"models/saved/generator_epoch{epoch}.pt")
        torch.save(discriminator.state_dict(), f"models/saved/discriminator_epoch{epoch}.pt")
