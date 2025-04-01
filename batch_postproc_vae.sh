#!/bin/bash
#SBATCH --job-name=postproc
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --time=0:30:00
#SBATCH --out=postproc.%j
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=EUHPC_B20_015
#SBATCH --mem=256000

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

source ../pyenvs/flowgen/bin/activate
#########################

# EXECUTION ########
 #mpirun -np $(($SLURM_NTASKS)) python evaluate_autoencoder.py --seq_len 5 100 --save_path experiments/postproc_vae \
 #       --data_path /scratch/cfd/gonzalez/HIT_LES_COMP/ --vae_params vae_config_8.yaml\
 #       --ckpt_path /scratch/cfd/gonzalez/flowgen/experiments/VAE-258/lightning_logs/version_1537905/checkpoints/epoch=1069-step=105472.ckpt

 srun  python evaluate_autoencoder.py --seq_len 5 100 --save_path experiments/postproc_vae \
        --data_path /leonardo_work/EUHPC_B20_015/HIT_LES_COMP/ --vae_params vae_config_8.yaml\
        --ckpt_path /leonardo_work/EUHPC_B20_015/flowgen/.aim/None/0b13b2bc49c84b0c92142ca4/checkpoints/epoch=1031-step=174408.ckpt
#########################