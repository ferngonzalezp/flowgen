#!/bin/bash
#SBATCH --job-name=TFNO_off
#SBATCH --partition=gpua30rk8
#SBATCH --nodes=2
#SBATCH --time=6:00:00
#SBATCH --output=offline.%j
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
#export PYTORCH_NO_CUDA_MEMORY_CACHING=1
module list

# EXTRA COMMANDS ########
source ../pyenvs/flowgen_rk8/bin/activate
#########################

# EXECUTION ########

#mpirun -np $(($SLURM_NTASKS)) python train_offline.py --save_path experiments --loss pushforward --devices 4 --nodes $SLURM_NNODES --batch_size 1 --lr 1e-3 --epochs 500 \
# --model TFNO_t   --data_path /scratch/cfd/gonzalez/HIT_LES_FORCED/ --cases case1 --seq_len 10 100 \
# --weight_decay 0.01 --accumulate_grad_batches $((64 / $SLURM_NTASKS)) --lr_warmup --lr_warmup_steps 1000  --overfit_batches 0 \
# --ckpt_path /scratch/cfd/gonzalez/flowgen/experiments/TFNO_t_pushforward-10/lightning_logs/version_1483725/checkpoints/epoch=376-step=13195.ckpt

 mpirun -np $(($SLURM_NTASKS)) python train_offline.py --save_path experiments --loss pushforward --devices 4 --nodes $SLURM_NNODES --batch_size 1 --lr 1e-3 --epochs 100 \
 --model TFNO_t   --data_path /scratch/cfd/gonzalez/HIT_LES_COMP/ --cases case1 case2 case3 --seq_len 10 100 \
 --weight_decay 0.01 --accumulate_grad_batches $((64 / $SLURM_NTASKS)) --lr_warmup --lr_warmup_steps 1000  --overfit_batches 0 \
 #--ckpt_path /scratch/cfd/gonzalez/flowgen/experiments/TFNO_t_pushforward-12/lightning_logs/version_1485082/checkpoints/epoch=11-step=1692.ckpt
#########################

