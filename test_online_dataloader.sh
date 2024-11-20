#!/bin/bash
cd simulations/HIT/
sbatch batch_test.sh
cd ../../
sbatch batch_test_dl.sh