import pytest
import torch
from data_loader import load_data, preprocess_data, create_dataloader

def test_load_data():
    # Assuming 'data/processed/train_data.pt' exists and is a valid path
    data = load_data('data/processed/train_data.pt')
    assert data is not None
    assert isinstance(data, pd.DataFrame)
    assert len(data) > 0

def test_preprocess_data():
    data = load_data('data/processed/train_data.pt')
    processed_data, scaler = preprocess_data(data)
    assert processed_data is not None
    assert processed_data.shape[1] > 0  # Ensure there are columns
    assert scaler is not None

def test_create_dataloader():
    data = load_data('data/processed/train_data.pt')
    processed_data, _ = preprocess_data(data)
    dataloader = create_dataloader(processed_data)
    assert isinstance(dataloader, torch.utils.data.DataLoader)
    assert len(dataloader) > 0  # Ensure there is at least one batch
