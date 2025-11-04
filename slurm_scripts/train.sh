#!/bin/bash

#SBATCH -J job_name        # Job name
#SBATCH -o name.o%j     # Name of stdout output file
#SBATCH -e name.e%j     # Name of stderr error file
#SBATCH -p gpu-a100              # Queue (partition) name
#SBATCH -N 1                     # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                 # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 00:05:00          # Run time (hh:mm:ss)
#SBATCH --mail-type=all      # Send email at begin and end of job
#SBATCH --mail-user=your_mail
#SBATCH -A your_allocation 


# assume data is already downloaded 
TMP_DIR="$SCRATCH"
DATASET_NAME="WaterDropSample"
DATA_PATH="your_data_path"
MODEL_PATH="${TMP_DIR}/${DATASET_NAME}/MLP/models/"
ROLLOUT_PATH="${TMP_DIR}/${DATASET_NAME}/MLP/rollout/"

# Train for a few steps.
NODEFILE=/tmp/hostfile
scontrol show hostnames  > $NODEFILE
NNODES=$(< $NODEFILE wc -l)
mpiexec.hydra -np $NNODES -ppn 1 ./slurm_scripts/train_gns_parallel.sh \
--data_path=${DATA_PATH} --model_path=${MODEL_PATH} --ntraining_steps=5 --mode='train' \
 --batch_size=1  --latent_dim=128 --nmlp_layers=1 --mlp_hidden_dim=128  \
--validation_interval=100 

