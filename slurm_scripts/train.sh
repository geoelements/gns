#!/bin/bash

#SBATCH -J MLP         # Job name
#SBATCH -o log/MLP.o%j     # Name of stdout output file
#SBATCH -e log/MLP.e%j     # Name of stderr error file
#SBATCH -p gpu-a100              # Queue (partition) name
#SBATCH -N 1                     # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                 # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 2:05:00          # Run time (hh:mm:ss)
#SBATCH --mail-type=all      # Send email at begin and end of job
#SBATCH --mail-user=sli@tacc.utexas.edu
#SBATCH -A ECS24003 


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
mpiexec.hydra -np $NNODES -ppn 1 /work/07980/sli4/ls6/gns/slurm_scripts/train_gns_parallel.sh \
--data_path=${DATA_PATH} --model_path=${MODEL_PATH} --ntraining_steps=10000 --mode='train' \
--use_kan=0 --batch_size=1
#--model_file=latest --train_state_file=latest


/opt/apps/tacc-apptainer/1.1.8/bin/apptainer exec --nv  \
/scratch/07980/sli4/containers/gns-test_latest.sif \
torchrun --standalone --nproc_per_node 1 gns/train_kan.py  \
--mode="rollout" --data_path=${DATA_PATH} --model_path=${MODEL_PATH} \
--model_file="model-10000.pt" --output_path=${ROLLOUT_PATH} --use_kan=0 

/opt/apps/tacc-apptainer/1.1.8/bin/apptainer exec --nv  \
/scratch/07980/sli4/containers/gns-test_latest.sif \
python3 -m gns.render_rollout --rollout_name=rollout_ex0 \
--rollout_dir=/scratch/07980/sli4/gns/WaterDropSample/MLP/rollout/

/opt/apps/tacc-apptainer/1.1.8/bin/apptainer exec --nv  \
/scratch/07980/sli4/containers/gns-test_latest.sif \
python3 -m gns.render_rollout --rollout_name=rollout_ex1 \
--rollout_dir=/scratch/07980/sli4/gns/WaterDropSample/MLP/rollout/
