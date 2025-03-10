#!/bin/bash
#SBATCH --job-name=postproc
#SBATCH --partition=gpua30rk8
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --output=postproc.%j
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
##SBATCH --reservation=gpua30-rocky8

# LOAD MODULES ##########
source /etc/profile.d/modules.sh
module purge 
module load python_rk8/anaconda3.10.9
module load  lib/phdf5/1.10.8_gcc112_ompi411 tools/cmake_rk8/3.31.0 lib/hdf5/1.10.8_gcc112
module load mpi/openmpi/4.1.1_gcc112
export PSM2_CUDA=0 #must be set to 0
export CC=mpicc
export CXX=mpiCC
export FC=mpif90
export PYTHONPATH="${PYTHONPATH}:/scratch/cfd/gonzalez/adios2_flowgen/cratch/cfd/gonzalez/pyenvs/flowgen_rk8/lib/python3.10/site-packages"
export LD_LIBRARY_PATH=/scratch/cfd/gonzalez/adios2_flowgen:$LD_LIBRARY_PATH
module list

# EXTRA COMMANDS ########
source ../pyenvs/flowgen_rk8/bin/activate
#########################

# EXECUTION ########
#mpirun -np $(($SLURM_NTASKS)) python postprocess.py --save_path experiments/post_proc --loss pushforward --devices 1 --nodes $SLURM_NNODES --batch_size 1 --lr 1e-3 --epochs 100 \
# --model UNET_shub  \
# --ckpt_path /scratch/cfd/gonzalez/flowgen/UNET_shub/Checkpoints_refined/UNET_TIME_Series_Pytorch_Lighting_Case_1_Complete_GPU_4_Seq_Lr_Sch_checkpoint-epoch=0399.ckpt \
# --unroll_steps 100 --data_path /scratch/cfd/gonzalez/HIT_LES_COMP/

mpirun -np $(($SLURM_NTASKS)) python postprocess.py --save_path experiments/post_proc --loss pushforward --devices 1 --nodes $SLURM_NNODES --batch_size 1 --lr 1e-3 --epochs 100 \
 --model TFNO_t  \
 --ckpt_path /scratch/cfd/gonzalez/flowgen/experiments/TFNO_t_pushforward-9/lightning_logs/version_1483453/checkpoints/epoch=152-step=5355.ckpt \
 --unroll_steps 1 --data_path /scratch/cfd/gonzalez/HIT_LES_FORCED/ --cases case1

#mpirun -np $(($SLURM_NTASKS)) python postprocess.py --save_path experiments/post_proc --loss pushforward --devices 1 --nodes $SLURM_NNODES --batch_size 1 --lr 1e-3 --epochs 100 \
# --model TFNO_t  \
# --ckpt_path /scratch/cfd/gonzalez/flowgen/experiments/TFNO_t_pushforward_online-5/lightning_logs/version_1483139/checkpoints/epoch=389-step=1560.ckpt \
# --unroll_steps 1 --data_path /scratch/cfd/gonzalez/HIT_LES_COMP/ --cases case1 case2 case3

#mpirun -np $(($SLURM_NTASKS)) python postprocess.py --save_path experiments/post_proc --loss pushforward --devices 1 --nodes $SLURM_NNODES --batch_size 1 --lr 1e-3 --epochs 200 \
# --model Attn_UNET  \
# --ckpt_path /scratch/cfd/gonzalez/flowgen/experiments/Attn_UNET_pushforward-40/lightning_logs/version_1475032/checkpoints/epoch=102-step=13613.ckpt \
# --unroll_steps 1 --data_path /scratch/cfd/gonzalez/HIT_LES_COMP/

#mpirun -np $(($SLURM_NTASKS)) python postprocess.py --save_path experiments/post_proc --loss pushforward --devices 1 --nodes $SLURM_NNODES --batch_size 1 --lr 1e-3 --epochs 100 \
# --model Attn_UNET  --dtype half \
# --ckpt_path /scratch/cfd/gonzalez/flowgen/experiments/Attn_UNET_pushforward-48/lightning_logs/version_1482246/checkpoints/epoch=128-step=4515.ckpt \
# --unroll_steps 100 --data_path /scratch/cfd/gonzalez/HIT_LES_FORCED/ --cases case1
#########################

