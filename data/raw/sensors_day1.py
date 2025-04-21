import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create directory if it doesn't exist
os.makedirs('data/raw', exist_ok=True)

# Parameters
num_minutes = 24 * 60  # one full day
start_time = datetime(2025, 4, 21, 0, 0, 0)

# Generate timestamps
timestamps = [start_time + timedelta(minutes=i) for i in range(num_minutes)]

# Generate sensor data
np.random.seed(42)  # reproducibility

temperature = np.random.normal(loc=75, scale=2, size=num_minutes)   # °C
pressure = np.random.normal(loc=5, scale=0.3, size=num_minutes)     # bar
flow_rate = np.random.normal(loc=100, scale=5, size=num_minutes)    # liters/sec
vibration = np.random.normal(loc=2, scale=0.1, size=num_minutes)    # mm/s

# Create DataFrame
df = pd.DataFrame({
    'timestamp': timestamps,
    'temperature_C': np.round(temperature, 2),
    'pressure_bar': np.round(pressure, 2),
    'flow_rate_lps': np.round(flow_rate, 2),
    'vibration_mms': np.round(vibration, 3)
})

# Save to CSV
csv_path = 'data/raw/sensors_day1.csv'
df.to_csv(csv_path, index=False)

print(f"✅ Synthetic sensor data saved to: {csv_path}")
