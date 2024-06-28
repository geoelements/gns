#!/bin/bash

LOCAL_RANK=$PMI_RANK

NODEFILE=/tmp/hostfile
scontrol show hostnames  > $NODEFILE
if [[ -z "${NODEFILE}" ]]; then
    RANKS=$NODEFILE
    NNODES=1
else
    MAIN_RANK=$(head -n 1 $NODEFILE)
    RANKS=$(tr '\n' ' ' < $NODEFILE)
    NNODES=$(< $NODEFILE wc -l)
fi


PRELOAD="/opt/apps/tacc-apptainer/1.1.8/bin/apptainer exec --nv $1 "
CMD="torchrun --nproc_per_node 4 --nnodes $NNODES --node_rank=$LOCAL_RANK --master_addr=$MAIN_RANK --master_port=1234 train.py"

FULL_CMD="$PRELOAD $CMD"
echo "Training command: $FULL_CMD"

eval $FULL_CMD