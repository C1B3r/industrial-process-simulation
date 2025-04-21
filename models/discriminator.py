import torch
import torch.nn as nn
import torch.optim as optim

# Dummy Generator and Discriminator models
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 10),  # Output size = sensor data features
            nn.Tanh()  # [-1, 1] range for normalized data
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Real or fake classification
        )

    def forward(self, x):
        return self.model(x)

# Initialize models
generator = Generator()
discriminator = Discriminator()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

# Loss function
criterion = nn.BCELoss()

# Dummy training loop (simplified)
epochs = 200
for epoch in range(1, epochs + 1):
    # Generate fake data
    z = torch.randn(64, 16)  # Latent vector size = 16
    fake_data = generator(z)

    # Real data (replace this with actual IoT data)
    real_data = torch.randn(64, 10)  # 10 sensor features

    # Labels
    real_labels = torch.ones(64, 1)
    fake_labels = torch.zeros(64, 1)

    # Train Discriminator
    discriminator.zero_grad()
    real_loss = criterion(discriminator(real_data), real_labels)
    fake_loss = criterion(discriminator(fake_data.detach()), fake_labels)
    d_loss = real_loss + fake_loss
    d_loss.backward()
    optimizer_D.step()

    # Train Generator
    generator.zero_grad()
    g_loss = criterion(discriminator(fake_data), real_labels)  # Fool the Discriminator
    g_loss.backward()
    optimizer_G.step()

    # Save model checkpoints every 100 epochs
    if epoch % 100 == 0:
        checkpoint_path_D = f"models/saved/discriminator_epoch{epoch}.pt"
        torch.save(discriminator.state_dict(), checkpoint_path_D)
        print(f"Discriminator checkpoint saved for epoch {epoch}")

    # Optionally, save generator checkpoint as well
    if epoch % 100 == 0:
        checkpoint_path_G = f"models/saved/generator_epoch{epoch}.pt"
        torch.save(generator.state_dict(), checkpoint_path_G)
        print(f"Generator checkpoint saved for epoch {epoch}")
