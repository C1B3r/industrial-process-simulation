import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Function to load raw data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Function to preprocess data (normalize)
def preprocess_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# Function to create PyTorch DataLoader for training and testing
def create_dataloader(data, batch_size=64):
    tensor_data = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
