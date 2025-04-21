import json
import torch
import torch.optim as optim
import torch.nn as nn

# Function to save training logs to a JSON file
def save_training_logs(logs, file_path='models/saved/training_logs.json'):
    with open(file_path, 'w') as f:
        json.dump(logs, f, indent=4)
    print(f"Training logs saved to {file_path}")

# Define your models (Generator and Discriminator)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Tanh()
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
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Initialize models
generator = Generator()
discriminator = Discriminator()

# Optimizers and loss function
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)
criterion = nn.BCELoss()

# Dummy training loop
epochs = 200
training_logs = []  # Initialize training log list

for epoch in range(1, epochs + 1):
    # Fake input data (latent vector)
    z = torch.randn(64, 16)
    fake_data = generator(z)

    # Real data (replace with actual data)
    real_data = torch.randn(64, 10)  # Fake data, 10 sensor features

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
    g_loss = criterion(discriminator(fake_data), real_labels)  # Try to fool the Discriminator
    g_loss.backward()
    optimizer_G.step()

    # Log training progress after each epoch
    epoch_log = {
        'epoch': epoch,
        'generator_loss': g_loss.item(),
        'discriminator_loss': d_loss.item(),
        'learning_rate_generator': optimizer_G.param_groups[0]['lr'],
        'learning_rate_discriminator': optimizer_D.param_groups[0]['lr']
    }

    training_logs.append(epoch_log)  # Append the log for this epoch

    # Optionally save the logs after every epoch
    save_training_logs(training_logs)

    # Save the models (checkpoints) as needed
    if epoch % 100 == 0:
        checkpoint_path_G = f"models/saved/generator_epoch{epoch}.pt"
        torch.save(generator.state_dict(), checkpoint_path_G)

        checkpoint_path_D = f"models/saved/discriminator_epoch{epoch}.pt"
        torch.save(discriminator.state_dict(), checkpoint_path_D)
