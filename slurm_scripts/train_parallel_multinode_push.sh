#!/bin/bash

#SBATCH -J job_name         # Job name
#SBATCH -o job_name.o%j     # Name of stdout output file
#SBATCH -e job_name.e%j     # Name of stderr error file
#SBATCH -p queue_name       # Queue (partition) name
#SBATCH -N num_nodes        # Total # of nodes (must be 1 for serial)
#SBATCH -n num_task_same_as_num_nodes    # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 00:01:00          # Run time (hh:mm:ss)
#SBATCH --mail-type=all      # Send email at begin and end of job
#SBATCH --mail-user=your_email
#SBATCH -A allocation_name 


# assume data is already downloaded 
TMP_DIR="$SCRATCH/gns/"
DATASET_NAME="WaterDropSample"
DATA_PATH="${TMP_DIR}/${DATASET_NAME}/dataset/"
MODEL_PATH="${TMP_DIR}/${DATASET_NAME}/models/"
ROLLOUT_PATH="${TMP_DIR}/${DATASET_NAME}/rollout/"

# Training params
USE_KAN=0 #0 for using MLP, 1 for using KAN

# Train for a few steps.
NODEFILE=/tmp/hostfile
scontrol show hostnames  > $NODEFILE
NNODES=$(< $NODEFILE wc -l)
mpiexec.hydra -np $NNODES -ppn 1 /work/07980/sli4/ls6/gns/slurm_scripts/train_gns_parallel.sh \
--data_path=${DATA_PATH} --model_path=${MODEL_PATH} --ntraining_steps=100 --mode='train' \
--use_kan=$USE_KAN 
