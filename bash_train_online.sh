#!/bin/bash
sbatch batch_train_online.sh
cd simulations/HIT/
rm -r  -f train_online/*.sst
sbatch batch_test.sh
cd ../../