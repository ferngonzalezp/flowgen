#!/bin/bash

# Check if config file is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_file>"
    echo "Example: $0 configs/experiment1.conf"
    exit 1
fi

CONFIG_FILE=$1

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found"
    exit 1
fi

# Load default values
NODES=1
TASKS_PER_NODE=4
CPUS_PER_TASK=8
GPUS=4
TIME="0:30:00"
MEM="400000"
GLOBAL_BATCH_SIZE=256
BATCH_SIZE=8
EPOCHS=100
LR=1e-3
WEIGHT_DECAY=0.01
WARMUP_STEPS=1000
LIMIT_TRAIN_BATCHES=1024
MODEL="TFNO_t"
LOSS="one_step"
DATA_PATH="/leonardo_work/EUHPC_B20_015/HIT_LES_COMP/"
CASES="case1 case2 case3"
SEQ_LEN="2 2"
CKPT_PATH=""
JOB_NAME="offline_training"
PARTITION="boost_usr_prod"
ACCOUNT="EUHPC_B20_015"
MODEL_CONFIG="TFNO_t_1.yaml"

# Source the config file to override defaults
source "$CONFIG_FILE"

# Create temporary sbatch file
TEMP_FILE=$(mktemp)

cat > $TEMP_FILE << EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=$PARTITION
#SBATCH --nodes=$NODES
#SBATCH --time=$TIME
#SBATCH --out=${JOB_NAME}.%j
#SBATCH --ntasks-per-node=$TASKS_PER_NODE
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --gres=gpu:$GPUS
#SBATCH --account=$ACCOUNT
#SBATCH --mem=$MEM

# LOAD MODULES ##########
module load py-mpi4py/3.1.4--openmpi--4.1.6--gcc--12.2.0 \\
            hdf5/1.14.3--gcc--12.2.0 \\
            cmake/3.27.7 \\
            cudnn/8.9.7.29-12--gcc--12.2.0-cuda-12.1 \\
            cuda/12.1
export PSM2_CUDA=0 #must be set to 0
export CC=mpicc
export CXX=mpiCC
export FC=mpif90
export PYTHONPATH="\${PYTHONPATH}:/leonardo_work/EUHPC_B20_015/adios2_flowgen/_work/EUHPC_B20_015/pyenvs/flowgen/lib/python3.11/site-packages"
export LD_LIBRARY_PATH=/leonardo_work/EUHPC_B20_015/adios2_flowgen:\$LD_LIBRARY_PATH


source ../pyenvs/flowgen/bin/activate
#########################
export GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE
export BATCH_SIZE=$BATCH_SIZE
export EFF_BATCH_SIZE=\$((\$SLURM_NTASKS*\$BATCH_SIZE))
#########################

# EXECUTION ########
srun python train_offline.py --save_path experiments --loss $LOSS --devices \$SLURM_NTASKS_PER_NODE --nodes \$SLURM_NNODES --batch_size \$BATCH_SIZE --lr $LR --epochs $EPOCHS \\
 --model $MODEL --data_path $DATA_PATH --cases $CASES --seq_len $SEQ_LEN \\
 --weight_decay $WEIGHT_DECAY --accumulate_grad_batches \$((\$GLOBAL_BATCH_SIZE / \$EFF_BATCH_SIZE)) --lr_warmup --lr_warmup_steps $WARMUP_STEPS --overfit_batches 0 \\
 --limit_train_batches $LIMIT_TRAIN_BATCHES --model_config $MODEL_CONFIG
EOF

# Add checkpoint path if provided
if [ ! -z "$CKPT_PATH" ]; then
  echo " --ckpt_path $CKPT_PATH" >> $TEMP_FILE
fi

echo "#########################" >> $TEMP_FILE

# Submit the job
JOB_ID=$(sbatch $TEMP_FILE | awk '{print $NF}')

# Clean up
rm $TEMP_FILE

echo "Job submitted with ID: $JOB_ID"
echo "Configuration from: $CONFIG_FILE"
echo "Nodes: $NODES"
echo "Tasks per node: $TASKS_PER_NODE"
echo "CPUs per task: $CPUS_PER_TASK"
echo "GPUs: $GPUS"
echo "Global batch size: $GLOBAL_BATCH_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Model: $MODEL"
echo "Loss: $LOSS" 