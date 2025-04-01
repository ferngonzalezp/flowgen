#!/bin/bash
# Start simulations first
cd simulations/HIT/
rm -rf train_online/*.sst
sbatch batch_run_stream.sh
cd ../../

# Wait for simulations to initialize
sleep 5  # Adjust based on your system

# Then start training
sbatch batch_train_online.sh