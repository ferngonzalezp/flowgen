#!/bin/bash
#SBATCH --job-name=test_online_tr
#SBATCH --partition=gpua30
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --output=output.%j
#SBATCH --ntasks-per-node=1
##SBATCH --exclusive
#SBATCH --gres=gpu:1

# LOAD MODULES ##########
module purge 
module load python/3.9.5 mpi/openmpi/4.1.1_gcc112 lib/hdf5/1.10.1_gcc tools/cmake/3.23.0
export PSM2_CUDA=0
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
mpirun -np 1 python test.py
#########################

