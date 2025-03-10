#!/bin/bash
#SBATCH --job-name=postproc_vae
#SBATCH --partition=gpua30
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --output=postproc_vae.%j
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# LOAD MODULES ##########
module purge 
module load python/anaconda3.10.9
module load  lib/phdf5/1.10.8_gcc112_ompi411 tools/cmake/3.31.0 lib/hdf5/1.10.8_gcc112
module load mpi/openmpi/4.1.1_gcc112
export PSM2_CUDA=0 #must be set to 0
export CC=mpicc
export CXX=mpiCC
export FC=mpif90
export PYTHONPATH="${PYTHONPATH}:/scratch/cfd/gonzalez/adios2_flowgen/cratch/cfd/gonzalez/pyenvs/flowgen_rk8/lib/python3.10/site-packages"
export LD_LIBRARY_PATH=/scratch/cfd/gonzalez/adios2_flowgen:$LD_LIBRARY_PATH
module list

# EXTRA COMMANDS ########
source ../pyenvs/flowgen_rk8/bin/activate
#########################

# EXECUTION ########
 mpirun -np $(($SLURM_NTASKS)) python evaluate_autoencoder.py --seq_len 5 100 --save_path experiments/postproc_vae \
        --data_path /scratch/cfd/gonzalez/HIT_LES_COMP/ --vae_params vae_config_8.yaml\
        --ckpt_path /scratch/cfd/gonzalez/flowgen/experiments/VAE-258/lightning_logs/version_1537905/checkpoints/epoch=1069-step=105472.ckpt
#########################