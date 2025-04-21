#!/bin/bash

# Loop over different batch sizes and learning rates for experimentation
for batch_size in 32 64 128
do
    for lr in 0.0001 0.0002 0.0005
    do
        echo "Starting training with batch_size=$batch_size, lr=$lr"
        
        python src/train.py --epochs 100 --batch_size $batch_size --learning_rate $lr

        # You can add other customizations or logging here
        echo "Training complete for batch_size=$batch_size, lr=$lr"
    done
done
