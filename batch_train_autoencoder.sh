#!/bin/bash
#SBATCH --job-name=VAE
#SBATCH --partition=gpua30
#SBATCH --nodes=4
#SBATCH --time=6:00:00
#SBATCH --output=VAE.%j
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --exclusive

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
export OMPI_MCA_orte_base_help_aggregate=0
export BATCH_SIZE=4

 mpirun -np $(($SLURM_NTASKS)) python train_autoencoder.py --save_path experiments --devices $SLURM_NTASKS_PER_NODE --nodes $SLURM_NNODES \
 --batch_size $BATCH_SIZE --lr 1e-4 --epochs 1500 \
 --data_path /scratch/cfd/gonzalez/HIT_LES_COMP/ --cases case1 case2 case3 --seq_len 5 5 --vae_params vae_config_8.yaml \
 --overfit_batches 0 --beta 0 --accumulate_grad_batches $((256 / ($SLURM_NTASKS * $BATCH_SIZE))) \
 --ckpt_path /scratch/cfd/gonzalez/flowgen/experiments/VAE-258/lightning_logs/version_1537905/checkpoints/epoch=1079-step=108852.ckpt
 #--fine_tune_recon --pre_trained_pth /scratch/cfd/gonzalez/flowgen/experiments/VAE-40/lightning_logs/version_1511174/checkpoints/epoch=74-step=42225.ckpt \
#########################

