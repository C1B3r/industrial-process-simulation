import sys
import os

# Add the root project directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models')))

import torch
import pandas as pd
import numpy as np
from generator import Generator  # Now import directly, as models is added to path
import os

# Path to save the generated data
output_dir = 'data/synthetic'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'generated_batch_01.csv')

# Load the Generator model (no pre-trained weights for now)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)

# Optionally, you can skip this step if you don't have a pre-trained model
# generator.load_state_dict(torch.load(checkpoint_path, map_location=device))

generator.eval()  # Set the model to evaluation mode

# Generate synthetic data (batch of 100 samples)
batch_size = 100
latent_dim = 16  # Should match the latent dimension used during training

# Generate random noise vector
z = torch.randn(batch_size, latent_dim).to(device)

# Generate synthetic sensor data
synthetic_data = generator(z).cpu().detach().numpy()

# Convert synthetic data into a Pandas DataFrame
column_names = [f"sensor_{i+1}" for i in range(synthetic_data.shape[1])]
df = pd.DataFrame(synthetic_data, columns=column_names)

# Save the generated data to CSV
df.to_csv(output_file, index=False)

print(f"Generated data saved to: {output_file}")
