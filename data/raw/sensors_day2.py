import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create directory if it doesn't exist
os.makedirs('data/raw', exist_ok=True)

# Parameters
num_minutes = 24 * 60  # one full day
start_time = datetime(2025, 4, 22, 0, 0, 0)  # Day 2

# Generate timestamps
timestamps = [start_time + timedelta(minutes=i) for i in range(num_minutes)]

# Generate sensor data for Day 2 with slight drift
np.random.seed(43)  # different seed for Day 2

temperature = np.random.normal(loc=76, scale=2.5, size=num_minutes)   # °C - slightly higher due to day conditions
pressure = np.random.normal(loc=5.1, scale=0.35, size=num_minutes)    # bar - slight shift
flow_rate = np.random.normal(loc=102, scale=5.5, size=num_minutes)    # liters/sec - higher load
vibration = np.random.normal(loc=2.1, scale=0.12, size=num_minutes)   # mm/s - small change in machine behavior

# Create DataFrame
df = pd.DataFrame({
    'timestamp': timestamps,
    'temperature_C': np.round(temperature, 2),
    'pressure_bar': np.round(pressure, 2),
    'flow_rate_lps': np.round(flow_rate, 2),
    'vibration_mms': np.round(vibration, 3)
})

# Save to CSV
csv_path = 'data/raw/sensors_day2.csv'
df.to_csv(csv_path, index=False)

print(f"✅ Synthetic sensor data for Day 2 saved to: {csv_path}")
