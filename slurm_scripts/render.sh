#!/bin/bash

#SBATCH -J pyt_render        # Job name
#SBATCH -o pyt_render.o%j    # Name of stdout output file
#SBATCH -e pyt_render.e%j    # Name of stderr error file
#SBATCH -p rtx               # Queue (partition) name
#SBATCH -N 1                 # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                 # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 15:00:00          # Run time (hh:mm:ss)
#SBATCH --mail-type=all      # Send email at begin and end of job
#SBATCH -A BCS20003          # Project/Allocation name (req'd if you have more than 1)

# fail on error
set -e

# start in slurm_scripts
cd ..
source start_venv.sh

# assume data is already downloaded and hardcode WaterDropSample
data="mpm-columns"
python3 -m gns.render_rollout \
--rollout_path="${SCRATCH}/gns_pytorch/${data}/rollouts/rollout_0.pkl"

