#!/bin/bash

#SBATCH -J split        # Job name
#SBATCH -o log/split-multi.o%j     # Name of stdout output file
#SBATCH -e log/split-multi.e%j     # Name of stderr error file
#SBATCH -p gpu-a100-dev            # Queue (partition) name
#SBATCH -N 1                     # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                 # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 00:05:00          # Run time (hh:mm:ss)
#SBATCH --mail-type=all      # Send email at begin and end of job
#SBATCH --mail-user=sli@tacc.utexas.edu
#SBATCH -A BCS20003 


# assume data is already downloaded 
TMP_DIR="$SCRATCH/gns/"
DATASET_NAME="WaterDropSample"
DATA_PATH="/scratch/07980/sli4/data/gns/PRJ-3702/WaterDropSample/dataset/"
MODEL_PATH="${TMP_DIR}/${DATASET_NAME}/MLP/models/"
ROLLOUT_PATH="${TMP_DIR}/${DATASET_NAME}/MLP/rollout/"

# Train for a few steps.
NODEFILE=/tmp/hostfile
scontrol show hostnames  > $NODEFILE
NNODES=$(< $NODEFILE wc -l)
mpiexec.hydra -np $NNODES -ppn 1 /work/07980/sli4/ls6/code/gns/slurm_scripts/train_gns_parallel.sh \
gns/train.py \
--data_path=${DATA_PATH} --model_path=${MODEL_PATH} --ntraining_steps=2 --mode='train' \
--latent_dim=128 --nmlp_layers=1 --mlp_hidden_dim=128  \
--validation_interval=100 #--model_file "latest" --train_state_file "latest" 

