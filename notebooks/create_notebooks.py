import nbformat as nbf
import os

# Folder path
notebook_dir = "notebooks"
os.makedirs(notebook_dir, exist_ok=True)

# Notebook templates
notebooks = {
    "01_data_exploration.ipynb": [
        "# Data Exploration\nThis notebook explores raw IoT sensor data.",
        "import pandas as pd\nimport matplotlib.pyplot as plt\n\n# Load your raw data\ndata = pd.read_csv('../data/raw/sensors_day1.csv')\ndata.head()",
        "data.describe()",
        "data.plot(subplots=True, figsize=(12, 8))\nplt.show()"
    ],
    "02_preprocessing_pipeline.ipynb": [
        "# Preprocessing Pipeline\nThis notebook preprocesses raw IoT sensor data for training.",
        "import pandas as pd\nfrom sklearn.preprocessing import StandardScaler\nimport pickle",
        "data = pd.read_csv('../data/raw/sensors_day1.csv')\nscaler = StandardScaler()\nscaled_data = scaler.fit_transform(data)\n\nwith open('../data/processed/scaler.pkl', 'wb') as f:\n    pickle.dump(scaler, f)"
    ],
    "03_gan_training_experiment.ipynb": [
        "# GAN Training Experiment\nPrototype for training a GAN to simulate sensor data.",
        "import torch\nimport torch.nn as nn\nimport torch.optim as optim",
        "# Define your Generator and Discriminator classes here\n# Train your models\nprint('Training loop placeholder')"
    ],
    "04_synthetic_data_analysis.ipynb": [
        "# Synthetic Data Analysis\nVisualizing and comparing real vs generated sensor data.",
        "import pandas as pd\nimport matplotlib.pyplot as plt",
        "real_data = pd.read_csv('../data/raw/sensors_day1.csv')\ngenerated_data = pd.read_csv('../data/synthetic/generated_batch_01.csv')",
        "real_data.hist(figsize=(12, 8))\nplt.suptitle('Real Sensor Data')\nplt.show()",
        "generated_data.hist(figsize=(12, 8))\nplt.suptitle('Generated Sensor Data')\nplt.show()"
    ]
}

# Create notebooks
for name, cells in notebooks.items():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(c) if i == 0 else nbf.v4.new_code_cell(c) for i, c in enumerate(cells)]
    path = os.path.join(notebook_dir, name)
    with open(path, 'w') as f:
        nbf.write(nb, f)

print("✅ Jupyter notebooks created successfully in the 'notebooks/' folder.")
