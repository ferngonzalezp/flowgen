#!/bin/bash
#SBATCH --job-name=offline
#SBATCH --partition=gpua30
#SBATCH --nodes=4
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

#mpirun -np $(($SLURM_NTASKS)) python train_offline.py --save_path experiments --loss pushforward --devices 4 --nodes $SLURM_NNODES --batch_size 1 --lr 1e-3 --epochs 200 \
# --model Attn_UNET --precision 16-mixed  --weight_decay 0.001 --accumulate_grad_batches $((64 / $SLURM_NTASKS)) \
# --data_path /scratch/cfd/gonzalez/HIT_LES_COMP/  --lr_warmup --lr_warmup_steps 1000  --overfit_batches 0 \
# --ckpt_path /scratch/cfd/gonzalez/flowgen/experiments/Attn_UNET_pushforward-40/lightning_logs/version_1475032/checkpoints/epoch=103-step=13754.ckpt

 mpirun -np $(($SLURM_NTASKS)) python train_offline.py --save_path experiments --loss pushforward --devices 4 --nodes $SLURM_NNODES --batch_size 1 --lr 1e-3 --epochs 500 \
 --model Attn_UNET   --precision 16-mixed  --weight_decay 0.001 --data_path /scratch/cfd/gonzalez/HIT_LES_FORCED/ --cases case1 --seq_len 10 100 \
 --accumulate_grad_batches $((64 / $SLURM_NTASKS)) --lr_warmup --lr_warmup_steps 1000  --overfit_batches 0 \
 --ckpt_path /scratch/cfd/gonzalez/flowgen/experiments/Attn_UNET_pushforward-48/lightning_logs/version_1482246/checkpoints/epoch=134-step=4725.ckpt
#########################

