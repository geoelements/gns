#!/bin/bash
echo "enter"
echo $PMI_RANK

LOCAL_RANK=$PMI_RANK
CMD="gns/train_multinode.py  $@"

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

PRELOAD="source start_venv.sh;"

LAUNCHER="python -m torch.distributed.launch "
LAUNCHER+="--nnodes=$NNODES  --nproc_per_node=$GPU_PER_NODE \
--node_rank=$LOCAL_RANK --master_addr=$MAIN_RANK --max_restarts=0 "

# Combine preload, launcher, and script+args into full command
FULL_CMD="$PRELOAD $LAUNCHER $CMD"

echo $FULL_CMD 

eval $FULL_CMD &

wait
