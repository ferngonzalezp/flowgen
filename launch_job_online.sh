#!/bin/bash

# Check if config files are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <simulation_config_file> <training_config_file>"
    echo "Example: $0 configs/sim_experiment1.conf configs/train_experiment1.conf"
    exit 1
fi

SIM_CONFIG_FILE=$1
TRAIN_CONFIG_FILE=$2

# Check if config files exist
if [ ! -f "$SIM_CONFIG_FILE" ]; then
    echo "Error: Simulation config file '$SIM_CONFIG_FILE' not found"
    exit 1
fi

if [ ! -f "$TRAIN_CONFIG_FILE" ]; then
    echo "Error: Training config file '$TRAIN_CONFIG_FILE' not found"
    exit 1
fi

# Load default values for simulation
SIM_PARTITION="boost_usr_prod"
SIM_QOS="boost_qos_dbg"
SIM_NODES=1
SIM_TIME="0:30:00"
SIM_NTASKS=1
SIM_CPUS_PER_TASK=32
SIM_ACCOUNT="EUHPC_B20_015"
SIM_MEM="32000"
SIM_JOB_NAME="simulation"
SIM_OUTPUT_DIR="train_online"
SIM_WAIT_TIME=5

# Source the simulation config file to override defaults
source "$SIM_CONFIG_FILE"

# Load default values for training
TRAIN_JOB_NAME="online_training"
TRAIN_PARTITION="boost_usr_prod"
TRAIN_QOS="boost_qos_dbg"
TRAIN_NODES=2
TRAIN_TIME="0:30:00"
TRAIN_TASKS_PER_NODE=4
TRAIN_CPUS_PER_TASK=8
TRAIN_GPUS=4
TRAIN_ACCOUNT="EUHPC_B20_015"
TRAIN_MEM="400000"
GLOBAL_BATCH_SIZE=256
BATCH_SIZE=8
STEPS=-1
LR=1e-3
WEIGHT_DECAY=0.01
WARMUP_STEPS=1000
LIMIT_TRAIN_BATCHES="$((25600 / \$EFF_BATCH_SIZE))"
MODEL="TFNO_t"
LOSS="one_step"
DATA_PATH="/leonardo_work/EUHPC_B20_015/HIT_LES_COMP/"
CASES="case1 case2 case3"
SEQ_LEN="2 2"
CKPT_PATH=""
STREAM_PATH="./simulations/HIT/train_online"
RESERVOIR_SIZE=300
N_STREAMS=1

# Source the training config file to override defaults
source "$TRAIN_CONFIG_FILE"

# Create temporary simulation sbatch file
SIM_TEMP_FILE=$(mktemp)

cat > $SIM_TEMP_FILE << EOF
#!/bin/bash
#SBATCH --job-name=$SIM_JOB_NAME
#SBATCH --partition=$SIM_PARTITION
#SBATCH --nodes=$SIM_NODES
#SBATCH --time=$SIM_TIME
#SBATCH --out=${SIM_JOB_NAME}.%j
#SBATCH --ntasks=$SIM_NTASKS
#SBATCH --cpus-per-task=$SIM_CPUS_PER_TASK
#SBATCH --account=$SIM_ACCOUNT
#SBATCH --mem=$SIM_MEM
#SBATCH --gres=gpu:$SIM_GPUS


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

# EXTRA COMMANDS ########
export SstVerbose=2
export FABRIC_IFACE=ib0
export FI_PSM2_DISCONNECT=1
export FI_OFI_RXM_USE_SRX=1
export FI_PROVIDER=tcp
export PYTHONFAULTHANDLER=2
source /leonardo_work/EUHPC_B20_015/pyenvs/flowgen/bin/activate
#########################

# Clean output directory
rm -rf $SIM_OUTPUT_DIR/*.sst

# EXECUTION ########
#cd simulations/HIT/
srun -n \$SLURM_NTASKS python run_multi_hit.py --realizations -1 --sim_directory $SIM_OUTPUT_DIR
#########################
EOF

# Create temporary training sbatch file
TRAIN_TEMP_FILE=$(mktemp)

cat > $TRAIN_TEMP_FILE << EOF
#!/bin/bash
#SBATCH --job-name=$TRAIN_JOB_NAME
#SBATCH --partition=$TRAIN_PARTITION
#SBATCH --nodes=$TRAIN_NODES
#SBATCH --time=$TRAIN_TIME
#SBATCH --out=${TRAIN_JOB_NAME}.%j
#SBATCH --ntasks-per-node=$TRAIN_TASKS_PER_NODE
#SBATCH --cpus-per-task=$TRAIN_CPUS_PER_TASK
#SBATCH --gres=gpu:$TRAIN_GPUS
#SBATCH --account=$TRAIN_ACCOUNT
#SBATCH --mem=$TRAIN_MEM

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

# EXTRA COMMANDS ########
export SstVerbose=2
export FABRIC_IFACE=ib0
export FI_PSM2_DISCONNECT=1
export FI_OFI_RXM_USE_SRX=1
export FI_PROVIDER=tcp
export PYTHONFAULTHANDLER=2
source ../pyenvs/flowgen/bin/activate
#########################
export GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE
export BATCH_SIZE=$BATCH_SIZE
export EFF_BATCH_SIZE=\$((\$SLURM_NTASKS*\$BATCH_SIZE))

# EXECUTION ########
srun python train_online.py --steps $STEPS --save_path experiments --loss $LOSS --devices \$SLURM_NTASKS_PER_NODE --nodes \$SLURM_NNODES --batch_size \$BATCH_SIZE --lr $LR \\
 --model $MODEL --data_path $DATA_PATH --cases $CASES --seq_len $SEQ_LEN \\
 --weight_decay $WEIGHT_DECAY --accumulate_grad_batches \$((\$GLOBAL_BATCH_SIZE / \$EFF_BATCH_SIZE)) \\
 --lr_warmup --lr_warmup_steps $WARMUP_STEPS --model_config $MODEL_CONFIG \\
 --stream_path $STREAM_PATH --reservoir_size $RESERVOIR_SIZE --n_streams $N_STREAMS --reservoir_per_node
EOF

# Add checkpoint path if provided
if [ ! -z "$CKPT_PATH" ]; then
  echo " --ckpt_path $CKPT_PATH" >> $TRAIN_TEMP_FILE
fi

echo "#########################" >> $TRAIN_TEMP_FILE

mkdir -p $STREAM_PATH
# Submit the simulation job
echo "Submitting simulation job..."
cd simulations/HIT
SIM_JOB_ID=$(sbatch $SIM_TEMP_FILE | awk '{print $NF}')
echo "Simulation job submitted with ID: $SIM_JOB_ID"
cd ../../

# Wait for simulation to initialize
echo "Waiting $SIM_WAIT_TIME seconds for simulation to initialize..."
sleep $SIM_WAIT_TIME

# Submit the training job
echo "Submitting training job..."
TRAIN_JOB_ID=$(sbatch $TRAIN_TEMP_FILE | awk '{print $NF}')
echo "Training job submitted with ID: $TRAIN_JOB_ID"

# Clean up
rm $SIM_TEMP_FILE
rm $TRAIN_TEMP_FILE

echo "Online training launched with configuration:"
echo "Simulation config: $SIM_CONFIG_FILE"
echo "Training config: $TRAIN_CONFIG_FILE"
echo "Simulation job ID: $SIM_JOB_ID"
echo "Training job ID: $TRAIN_JOB_ID" 