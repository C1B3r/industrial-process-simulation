import torch
import pandas as pd
from models.generator import Generator

# Load trained generator model
generator = Generator()
generator.load_state_dict(torch.load('models/saved/generator_epoch100.pt'))
generator.eval()

# Generate synthetic data
latent_dim = 16
num_samples = 1000
z = torch.randn(num_samples, latent_dim)
synthetic_data = generator(z)

# Convert to DataFrame for easier analysis or saving
synthetic_data_df = pd.DataFrame(synthetic_data.detach().numpy(), columns=['sensor_' + str(i) for i in range(1, 11)])

# Save the synthetic data
synthetic_data_df.to_csv('data/synthetic/generated_batch_01.csv', index=False)

print("Synthetic data generated and saved!")
