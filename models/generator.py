import torch
import torch.nn as nn
import torch.optim as optim

# Dummy Generator Model
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

# Initialize Generator model
generator = Generator()

# Dummy Training Loop (for demonstration)
epochs = 200
optimizer = optim.Adam(generator.parameters(), lr=0.0002)

# Save checkpoint every 100 epochs
for epoch in range(1, epochs + 1):
    # Fake input data (latent vector)
    z = torch.randn(64, 16)  # Batch size 64, latent vector size 16
    generated_data = generator(z)  # Generate data

    # Optimize model here (skipping actual training code for brevity)
    optimizer.zero_grad()
    # Normally you'd compute loss and call loss.backward() here, but we'll skip it for now
    
    # Save model checkpoint after every 100 epochs
    if epoch % 100 == 0:
        checkpoint_path = f"models/saved/generator_epoch{epoch}.pt"
        torch.save(generator.state_dict(), checkpoint_path)
        print(f"Checkpoint saved for epoch {epoch}")

# Load the model after training
checkpoint_path = "models/saved/generator_epoch100.pt"
generator.load_state_dict(torch.load(checkpoint_path))
generator.eval()  # Set the model to evaluation mode

print("Model loaded and ready for inference!")
