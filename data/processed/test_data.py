import pandas as pd

def preprocess_data(raw_data):
    # Convert timestamp columns or non-numeric columns to string or datetime if necessary
    # For simplicity, let's assume the timestamp column is the first column (index 0)
    raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'], errors='coerce')

    # Drop the timestamp column or any non-numeric columns if necessary
    numeric_data = raw_data.select_dtypes(include=[float, int])
    
    # Fill missing values with the column mean for numeric data
    data_cleaned = numeric_data.fillna(numeric_data.mean())
    
    return data_cleaned

# Usage of the function:
def main():
    # Load your data (adjust the path as necessary)
    raw_data = pd.read_csv('data/raw/sensors_day1.csv')  # Replace with your actual path
    
    print(f"Raw data loaded successfully with shape: {raw_data.shape}")
    
    # Preprocess data
    processed_data = preprocess_data(raw_data)
    print(f"Processed data shape: {processed_data.shape}")
    
    # Save the processed data
    processed_data.to_pickle("test_data.pt")
    print("Processed data saved successfully as test_data.pt")

if __name__ == "__main__":
    main()
