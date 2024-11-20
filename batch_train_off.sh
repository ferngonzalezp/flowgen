#!/bin/bash
#SBATCH --job-name=offline
#SBATCH --partition=gpua30
#SBATCH --nodes=2
#SBATCH --time=6:00:00
#SBATCH --output=offline.%j
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --exclusive

# LOAD MODULES ##########
module purge 
module load lib/hdf5/1.10.1_gcc tools/cmake/3.23.0
module load mpi/openmpi/4.1.1_gcc112
export PSM2_CUDA=0 #must be set to 0
export CC=mpicc
export CXX=mpiCC
export FC=mpif90
export PYTHONPATH="${PYTHONPATH}:/scratch/cfd/gonzalez/adios2_flowgen/cratch/cfd/gonzalez/pyenvs/flowgen/lib/python3.9/site-packages"
export LD_LIBRARY_PATH=/scratch/cfd/gonzalez/adios2_flowgen:$LD_LIBRARY_PATH
#export PYTORCH_NO_CUDA_MEMORY_CACHING=1
module list

# EXTRA COMMANDS ########
source ../pyenvs/flowgen/bin/activate
#########################

# EXECUTION ########

mpirun -np $(($SLURM_NTASKS)) python train_offline.py --save_path experiments --loss pushforward --devices 4 --nodes $SLURM_NNODES --batch_size 1 --lr 1e-4 --epochs 100 \
 --model TFNO_t   --data_path /scratch/cfd/gonzalez/HIT_LES_COMP/ \
 --ckpt_path /scratch/cfd/gonzalez/flowgen/experiments/TFNO_t_pushforward-1/lightning_logs/version_1468958/checkpoints/epoch=97-step=13818.ckpt
#########################

