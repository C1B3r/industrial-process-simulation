# Industrial Process Simulation with GANs

This project uses Generative Adversarial Networks (GANs) to simulate complex industrial processes, enabling predictive maintenance and process optimization. The generated synthetic data can be used for process analysis, anomaly detection, and system optimization.

## Project Structure

- `data/`: Contains raw and processed data files.
- `models/`: Defines the architecture for the Generator and Discriminator models.
- `scripts/`: Automation scripts for training, generating data, and batch processing.
- `src/`: Core code for training, evaluating, and processing data.
- `test/`: Unit tests to validate the codebase.
- `requirements.txt`: Python package dependencies.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/neavdak/industrial-process-simulation
    cd industrial-process-simulation
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Training the model:**
    Run the training loop to start the GAN training process.
    ```bash
    python src/train.py --epochs 100 --batch_size 64 --learning_rate 0.0002
    ```

2. **Generating synthetic data:**
    Use the trained generator model to generate synthetic sensor data.
    ```bash
    python scripts/generate_data.py
    ```

## Tests

To run unit tests for the project:
```bash
pytest test/
