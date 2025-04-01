#!/bin/bash
#SBATCH --job-name=offline_training
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --time=0:30:00
#SBATCH --out=offline.%j
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --account=EUHPC_B20_015
#SBATCH --mem=400000

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
export GLOBAL_BATCH_SIZE=256
export BATCH_SIZE=8
export EFF_BATCH_SIZE=$(($SLURM_NTASKS*$BATCH_SIZE))
#########################

# EXECUTION ########

#mpirun -np $(($SLURM_NTASKS)) python train_offline.py --save_path experiments --loss pushforward --devices 4 --nodes $SLURM_NNODES --batch_size 1 --lr 1e-3 --epochs 500 \
# --model TFNO_t   --data_path /scratch/cfd/gonzalez/HIT_LES_FORCED/ --cases case1 --seq_len 10 100 \
# --weight_decay 0.01 --accumulate_grad_batches $((64 / $SLURM_NTASKS)) --lr_warmup --lr_warmup_steps 1000  --overfit_batches 0 \
# --ckpt_path /scratch/cfd/gonzalez/flowgen/experiments/TFNO_t_pushforward-10/lightning_logs/version_1483725/checkpoints/epoch=376-step=13195.ckpt

 #mpirun -np $(($SLURM_NTASKS)) python train_offline.py --save_path experiments --loss pushforward --devices 4 --nodes $SLURM_NNODES --batch_size 1 --lr 1e-3 --epochs 100 \
 #--model TFNO_t   --data_path /scratch/cfd/gonzalez/HIT_LES_COMP/ --cases case1 case2 case3 --seq_len 10 100 \
 #-weight_decay 0.01 --accumulate_grad_batches $((64 / $SLURM_NTASKS)) --lr_warmup --lr_warmup_steps 1000  --overfit_batches 0 \
 #--ckpt_path /scratch/cfd/gonzalez/flowgen/experiments/TFNO_t_pushforward-12/lightning_logs/version_1485082/checkpoints/epoch=11-step=1692.ckpt

srun python train_offline.py --save_path experiments --loss one_step --devices $SLURM_NTASKS_PER_NODE --nodes $SLURM_NNODES --batch_size $BATCH_SIZE --lr 1e-3 --epochs 1 \
 --model TFNO_t   --data_path /leonardo_work/EUHPC_B20_015/HIT_LES_COMP/ --cases case1 case2 case3 --seq_len 2 2 \
 --weight_decay 0.01 --accumulate_grad_batches $(($GLOBAL_BATCH_SIZE / $EFF_BATCH_SIZE)) --lr_warmup --lr_warmup_steps 1000  --overfit_batches 0 \
 --limit_train_batches 1024 --model_config TFNO_t_5.yaml
 #--ckpt_path /scratch/cfd/gonzalez/flowgen/experiments/TFNO_t_pushforward-12/lightning_logs/version_1485082/checkpoints/epoch=11-step=1692.ckpt
#########################

