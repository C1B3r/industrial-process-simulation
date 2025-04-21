readme_content = """
# Models Folder

This directory contains the model architectures and saved weights for the Generative AI system used in Industrial Process Simulation.

---

## 📌 Generator (`generator.py`)

The Generator is a neural network that creates synthetic sensor data from random noise (latent vectors).

- **Input:** Latent vector of size 16.
- **Architecture:** Fully connected layers with ReLU activations.
- **Output:** Simulated sensor data with 10 features, scaled using `Tanh` activation for output in the range [-1, 1].
- **Purpose:** Generates realistic sensor data for predictive maintenance, digital twin testing, and process optimization.

---

## 📌 Discriminator (`discriminator.py`)

The Discriminator is a neural network that evaluates whether the input sensor data is real or generated.

- **Input:** Sensor data with 10 features.
- **Architecture:** Fully connected layers with LeakyReLU activations.
- **Output:** A single value between 0 and 1, representing the probability of the input being real (`1`) or fake (`0`).
- **Purpose:** Guides the Generator to produce realistic data by providing feedback during training.

---

## 💾 Saved Models (`saved/`)

This subdirectory contains model checkpoints saved during training:

- `generator_epochXXX.pt`: Saved Generator model at a specific training epoch.
- `discriminator_epochXXX.pt`: Saved Discriminator model at a specific training epoch.
- `training_logs.json`: Records training configuration, loss history, and evaluation metrics.

---

## ⚙️ Usage

To save and load models:

```python
# Saving
torch.save(generator.state_dict(), 'models/saved/generator_epoch100.pt')
torch.save(discriminator.state_dict(), 'models/saved/discriminator_epoch100.pt')

# Loading
generator.load_state_dict(torch.load('models/saved/generator_epoch100.pt'))
discriminator.load_state_dict(torch.load('models/saved/discriminator_epoch100.pt'))
