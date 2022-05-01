#!/bin/bash

#SBATCH -J pyt_mpm_train         # Job name
#SBATCH -o pyt_mpm_train.o%j     # Name of stdout output file
#SBATCH -e pyt_mpm_train.e%j     # Name of stderr error file
#SBATCH -p gpu-a100          # Queue (partition) name
#SBATCH -N 1                 # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                 # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 48:00:00          # Run time (hh:mm:ss)
#SBATCH --mail-type=all      # Send email at begin and end of job
#SBATCH -A OTH21021          # Project/Allocation name (req'd if you have more than 1)

# fail on error
set -e

# start in slurm_scripts
cd ..
source start_venv.sh

# assume data is already downloaded and hardcode WaterDropSample
data="mpm-columns"
python3 -m gns.train --data_path="${SCRATCH}/gns_pytorch/${data}/dataset/" \
--model_path="${SCRATCH}/gns_pytorch/${data}/models/" \
--output_path="${SCRATCH}/gns_pytorch/${data}/rollouts/" \
--cuda_device_number=0 \
--ntraining_steps=1000000
#--model_file="latest" \
#--train_state_file="latest"
