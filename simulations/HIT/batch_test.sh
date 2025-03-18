#!/bin/bash
#SBATCH --job-name=run_HIT
#SBATCH --partition=gpua30
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --output=output.%j
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

# LOAD MODULES ##########
module purge 
module load python/anaconda3.10.9
module load tools/cmake/3.31.0 lib/hdf5/1.10.8_gcc112
module load mpi/openmpi/4.1.1_gcc112
export PSM2_CUDA=0 #must be set to 0
export CC=mpicc
export CXX=mpiCC
export FC=mpif90
export PYTHONPATH="${PYTHONPATH}:/scratch/cfd/gonzalez/adios2_flowgen/cratch/cfd/gonzalez/pyenvs/flowgen_rk8/lib/python3.10/site-packages"
export LD_LIBRARY_PATH=/scratch/cfd/gonzalez/adios2_flowgen:$LD_LIBRARY_PATH
module list

# EXTRA COMMANDS ########
export SstVerbose=2
export FABRIC_IFACE=ib0
export FI_PSM2_DISCONNECT=1
export FI_OFI_RXM_USE_SRX=1
export FI_PROVIDER=tcp
source /scratch/cfd/gonzalez/pyenvs/flowgen_rk8/bin/activate
#########################

# EXECUTION ########
mpirun -np 1 python run_forced_hit.py --case_json /scratch/cfd/gonzalez/flowgen/simulations/HIT/HIT_decay_ma0.2.json
#########################

