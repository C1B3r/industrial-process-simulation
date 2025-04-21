import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Load the raw data
def load_raw_data(file_path):
    """
    Loads the raw CSV data.
    """
    data = pd.read_csv(file_path)
    return data

# Preprocess data
def preprocess_data(data):
    """
    Clean and preprocess the raw data.
    - Remove non-numeric columns (e.g., timestamps).
    - Normalize sensor readings using StandardScaler.
    """
    # Example: Identify and remove non-numeric columns (e.g., timestamp)
    non_numeric_cols = data.select_dtypes(exclude=['number']).columns
    data = data.drop(columns=non_numeric_cols)  # Remove non-numeric columns

    # Example: Drop rows with missing values
    data = data.dropna()

    # Normalize using StandardScaler (assuming all columns are sensor readings)
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)

    return data_normalized, scaler

# Save the processed data into a PyTorch tensor
def save_processed_data(data, scaler, output_path, scaler_path):
    """
    Save the processed data into a PyTorch tensor file and the scaler for future use.
    """
    # Convert to PyTorch tensor
    data_tensor = torch.tensor(data, dtype=torch.float32)
    
    # Save the tensor to .pt file
    torch.save(data_tensor, output_path)
    
    # Save the scaler to disk for later use
    import pickle
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Data saved as {output_path} and scaler saved as {scaler_path}")

def main():
    # Define the file paths
    raw_data_path = 'data/raw/sensors_day1.csv'  # Adjust as needed
    processed_data_path = 'data/processed/train_data.pt'
    scaler_file_path = 'data/processed/scaler.pkl'

    # Step 1: Load raw data
    print("Loading raw data...")
    raw_data = load_raw_data(raw_data_path)

    # Step 2: Preprocess the data
    print("Preprocessing data...")
    processed_data, scaler = preprocess_data(raw_data)

    # Step 3: Save processed data and scaler
    print("Saving processed data and scaler...")
    save_processed_data(processed_data, scaler, processed_data_path, scaler_file_path)

if __name__ == "__main__":
    main()
