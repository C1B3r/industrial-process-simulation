# Tests Overview

This folder contains unit tests for various modules in the industrial process simulation project.

- `test_data_loader.py`: Tests for data loading, preprocessing, and batching.
- `test_models.py`: Tests for the GAN model (Generator and Discriminator).
- `test_training.py`: Tests for the training loop and optimization.
- `test_evaluation.py`: Tests for synthetic data generation and evaluation.

## Running the tests:

```bash
# To run all tests
pytest test/

# To run specific tests
pytest test/test_models.py
