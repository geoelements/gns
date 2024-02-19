#!/bin/bash

#SBATCH -J multinode         # Job name
#SBATCH -o log/multinode.o%j     # Name of stdout output file
#SBATCH -e log/multinode.e%j     # Name of stderr error file
#SBATCH -p gpu-a100              # Queue (partition) name
#SBATCH -N 2                     # Total # of nodes (must be 1 for serial)
#SBATCH -n 2                 # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 0:05:00          # Run time (hh:mm:ss)
#SBATCH --mail-type=all      # Send email at begin and end of job

# fail on error
set -e

# start in slurm_scripts
#cd ..
source start_venv.sh

# assume data is already downloaded and hardcode WaterDropSample
TMP_DIR="./gns_sample"
DATASET_NAME="WaterDropSample"
DATA_PATH="${TMP_DIR}/${DATASET_NAME}/dataset/"
MODEL_PATH="${TMP_DIR}/${DATASET_NAME}/models/"
ROLLOUT_PATH="${TMP_DIR}/${DATASET_NAME}/rollout/"

# Train for a few steps.
NODEFILE=/tmp/hostfile
scontrol show hostnames  > $NODEFILE
NNODES=$(< $NODEFILE wc -l)
mpiexec.hydra -np $NNODES -ppn 1 /work/07980/sli4/ls6/gns/slurm_scripts/train_gns_parallel_push.sh --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --ntraining_steps=10000 --mode='train'
