#!/bin/bash
#SBATCH --job-name=online_HIT
#SBATCH --partition=gpua30
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --output=pf.%j
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --exclusive

# LOAD MODULES ##########
module purge 
module load lib/hdf5/1.10.1_gcc tools/cmake/3.23.0
module load mpi/openmpi/4.1.1_gcc94
export PSM2_CUDA=0 #must be set to 0
export CC=mpicc
export CXX=mpiCC
export FC=mpif90
export PYTHONPATH="${PYTHONPATH}:/scratch/cfd/gonzalez/adios2_flowgen/cratch/cfd/gonzalez/pyenvs/flowgen/lib/python3.9/site-packages"
export LD_LIBRARY_PATH=/scratch/cfd/gonzalez/adios2_flowgen:$LD_LIBRARY_PATH
module list

# EXTRA COMMANDS ########
export SstVerbose=2
export FABRIC_IFACE=ib0
export FI_PSM2_DISCONNECT=1
export FI_OFI_RXM_USE_SRX=1
export FI_PROVIDER=tcp
export PYTHONFAULTHANDLER=2
source ../pyenvs/flowgen/bin/activate
#########################
# EXECUTION ########
mpirun -np 4 python train.py --loss pushforward --devices 4 --batch_size 1 --lr 1e-3
#########################

