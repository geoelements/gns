#!/bin/bash

#SBATCH -J pyt_train_both         # Job name
#SBATCH -o pyt_train_both.o%j     # Name of stdout output file
#SBATCH -e pyt_train_both.e%j     # Name of stderr error file
#SBATCH -p gpu-a100               # Queue (partition) name
#SBATCH -N 1                 # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                 # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 48:00:00          # Run time (hh:mm:ss)
#SBATCH --mail-type=all      # Send email at begin and end of job
#SBATCH --mail-user=jvantassel@tacc.utexas.edu
#SBATCH -A OTH21021          # Project/Allocation name (req'd if you have more than 1)

# fail on error
set -e

# start in slurm_scripts
cd ..
source start_venv.sh

# assume data is already downloaded
data="mpm-columns"
python3 -m gns.train --data_path="${SCRATCH}/gns_pytorch/${data}/dataset/" \
--model_path="${SCRATCH}/gns_pytorch/${data}/models/" \
--output_path="${SCRATCH}/gns_pytorch/${data}/rollouts/" \
--ntraining_steps=1000000 \
--cuda_device_number=0 &
#--model_file="latest" \
#--train_state_file="latest" \
#--cuda_device_number=0 &


data="WaterDropSamplePytorch"
python3 -m gns.train --data_path="${SCRATCH}/gns_pytorch/${data}/dataset/" \
--model_path="${SCRATCH}/gns_pytorch/${data}/models/" \
--output_path="${SCRATCH}/gns_pytorch/${data}/rollouts/" \
--ntraining_steps=1000000 \
--cuda_device_number=1
#--model_file="latest" \
#--train_state_file="latest" \
#--cuda_device_number=1

