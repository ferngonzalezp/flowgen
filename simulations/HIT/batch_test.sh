#!/bin/bash
#SBATCH --job-name=run_HIT
#SBATCH --partition=gpua30
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --output=output.%j
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

# LOAD MODULES ##########
module load py-mpi4py/3.1.4--openmpi--4.1.6--gcc--12.2.0 \
            hdf5/1.14.3--gcc--12.2.0 \
            cmake/3.27.7 \
            cudnn/8.9.7.29-12--gcc--12.2.0-cuda-12.1 \
            cuda/12.1
export PSM2_CUDA=0 #must be set to 0
export CC=mpicc
export CXX=mpiCC
export FC=mpif90
export PYTHONPATH="${PYTHONPATH}:/leonardo_work/EUHPC_B20_015/adios2_flowgen/_work/EUHPC_B20_015/pyenvs/flowgen/lib/python3.11/site-packages"
export LD_LIBRARY_PATH=/leonardo_work/EUHPC_B20_015/adios2_flowgen:$LD_LIBRARY_PATH
module list

# EXTRA COMMANDS ########
export SstVerbose=2
export FABRIC_IFACE=ib0
export FI_PSM2_DISCONNECT=1
export FI_OFI_RXM_USE_SRX=1
export FI_PROVIDER=tcp
source /leonardo_work/EUHPC_B20_015/pyenvs/flowgen/bin/activate
#########################

# EXECUTION ########
mpirun -np 1 python run_forced_hit.py
#########################

