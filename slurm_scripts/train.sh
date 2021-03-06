#!/bin/bash

#SBATCH -J pyt_sand3d_train         # Job name
#SBATCH -o pyt_sand3d_train.o%j     # Name of stdout output file
#SBATCH -e pyt_sand3d_train.e%j     # Name of stderr error file
#SBATCH -p gpu-a100              # Queue (partition) name
#SBATCH -N 1                     # Total # of nodes (must be 1 for serial)
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

# assume data is already downloaded and hardcode WaterDropSample
data="Sand-3D"
python3 -m gns.train --data_path="${SCRATCH}/gns_pytorch/${data}/dataset/" \
--model_path="${SCRATCH}/gns_pytorch/${data}/models/" \
--output_path="${SCRATCH}/gns_pytorch/${data}/rollouts/" \
--nsave_steps=10000 \
--cuda_device_number=0 \
--ntraining_steps=5000000 \
--model_file="latest" \
--train_state_file="latest"
