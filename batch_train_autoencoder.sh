#!/bin/bash
#SBATCH --job-name=VAE
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --out=VAE.%j
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --account=EUHPC_B20_015
#SBATCH --mem=282000

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

# EXTRA COMMANDS ########
source ../pyenvs/flowgen/bin/activate
#########################

# EXECUTION ########
#export OMPI_MCA_orte_base_help_aggregate=0
export BATCH_SIZE=8

 #mpirun -np $(($SLURM_NTASKS)) python train_autoencoder.py --save_path experiments --devices $SLURM_NTASKS_PER_NODE --nodes $SLURM_NNODES \
 #--batch_size $BATCH_SIZE --lr 5e-4 --epochs 500 \
 #--data_path /leonardo_work/EUHPC_B20_015/HIT_LES_COMP --cases case1 case2 case3 --seq_len 5 5 --vae_params vae_config_8.yaml \
 #--overfit_batches 0 --beta 0 --accumulate_grad_batches $((256 / ($SLURM_NTASKS * $BATCH_SIZE))) \
 #--ckpt_path /scratch/cfd/gonzalez/flowgen/experiments/VAE-258/lightning_logs/version_1537905/checkpoints/epoch=1079-step=108852.ckpt
 #--fine_tune_recon --pre_trained_pth /scratch/cfd/gonzalez/flowgen/experiments/VAE-40/lightning_logs/version_1511174/checkpoints/epoch=74-step=42225.ckpt \

srun python train_autoencoder.py --save_path experiments --devices $SLURM_NTASKS_PER_NODE --nodes $SLURM_NNODES \
 --batch_size $BATCH_SIZE --lr 5e-4 --epochs 500 \
 --data_path /leonardo_work/EUHPC_B20_015/HIT_LES_COMP --cases case1 case2 case3 --seq_len 5 5 --vae_params vae_config_8.yaml \
 --overfit_batches 100 --beta 0 --accumulate_grad_batches $((256 / ($SLURM_NTASKS * $BATCH_SIZE))) \
#########################

