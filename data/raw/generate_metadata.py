import json
from datetime import datetime

# Example metadata dictionary
metadata = {
    "dataset_name": "Industrial Process Sensor Data",
    "description": "Raw sensor data collected from industrial machines for predictive maintenance and process optimization.",
    "created_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "sensors": [
        {"name": "Temperature", "unit": "Celsius", "range": [-40, 150]},
        {"name": "Pressure", "unit": "Bar", "range": [0, 300]},
        {"name": "FlowRate", "unit": "L/min", "range": [0, 500]},
        {"name": "Vibration", "unit": "mm/s", "range": [0, 50]},
        {"name": "Humidity", "unit": "%", "range": [0, 100]}
    ],
    "machines": [
        {"id": "MX-1001", "type": "Centrifugal Pump"},
        {"id": "MX-1002", "type": "Heat Exchanger"},
        {"id": "MX-1003", "type": "Compressor"}
    ],
    "collection_frequency": "1 sample/sec",
    "data_source": "IoT edge devices via MQTT",
    "author": "Your Company Name"
}

# Save metadata as JSON
output_path = "./data/raw/metadata.json"

# Writing to file
with open(output_path, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"metadata.json generated at {output_path}")
