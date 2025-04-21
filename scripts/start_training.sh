#!/bin/bash

# Set up environment (optional)
export PYTHONPATH=$(pwd)

# Run the training script
python src/train.py --epochs 100 --batch_size 64 --learning_rate 0.0002

# You can add other options or configurations here
echo "Training complete!"
