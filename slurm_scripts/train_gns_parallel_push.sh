#!/bin/bash
echo $PMI_RANK

LOCAL_RANK=$PMI_RANK
CMD="gns/train_kan.py  $@"

NODEFILE=/tmp/hostfile
scontrol show hostnames  > $NODEFILE


GPU_PER_NODE=$(nvidia-smi --list-gpus | wc -l)

if [[ -z "${NODEFILE}" ]]; then
    RANKS=$NODEFILE
    NNODES=1
else
    MAIN_RANK=$(head -n 1 $NODEFILE)
    RANKS=$(tr '\n' ' ' < $NODEFILE)
    NNODES=$(< $NODEFILE wc -l)
fi

PRELOAD="/opt/apps/tacc-apptainer/1.1.8/bin/apptainer exec --nv  /path/to/container "

LAUNCHER="torchrun "
LAUNCHER+="--nnodes=$NNODES  --nproc_per_node=$GPU_PER_NODE \
--node_rank=$LOCAL_RANK --master_addr=$MAIN_RANK --master_port=1234 --max_restarts=0 "

# Combine preload, launcher, and script+args into full command
FULL_CMD="$PRELOAD $LAUNCHER $CMD"

echo $FULL_CMD 

eval $FULL_CMD &

wait
