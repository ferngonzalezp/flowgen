#!/bin/bash
sbatch batch_test_dl.sh
cd simulations/HIT/
sbatch batch_test.sh
cd ../../