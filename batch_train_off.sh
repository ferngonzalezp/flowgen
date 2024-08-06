#!/bin/bash
#SBATCH --job-name=offline
#SBATCH --partition=gpua30
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --output=offline.%j
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclusive

# LOAD MODULES ##########
module purge 
module load lib/hdf5/1.10.1_gcc tools/cmake/3.23.0
module load mpi/openmpi/4.1.1_gcc112
export PSM2_CUDA=0 #must be set to 0
export CC=mpicc
export CXX=mpiCC
export FC=mpif90
module list

# EXTRA COMMANDS ########
source ../pyenvs/flowgen/bin/activate
#########################

# EXECUTION ########
mpirun -np $(($SLURM_NTASKS)) python train_offline.py --save_path experiments --loss pushforward --devices 1 --nodes $SLURM_NNODES --batch_size 1 --lr 1e-3 --epochs 1 --model TFNO_t  \
 #--ckpt_path /scratch/cfd/gonzalez/HIT_online_learning2/lightning_logs/version_1410081/checkpoints/epoch=23-step=6768.ckpt
#########################

