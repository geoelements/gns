#!/bin/bash

module reset

# start env
# ---------
ml cuda/11.3
ml cudnn
ml nccl

module load phdf5
module load python3/3.9
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

python3 -m virtualenv venv

source venv/bin/activate

# test env
# --------
echo 'which python -> venv'
which python

echo 'test_pytorch.py -> random tensor'
python test/test_pytorch.py 

echo 'test_pytorch_cuda_gpu.py -> True if GPU'
python test/test_pytorch_cuda_gpu.py

echo 'test_torch_geometric.py -> no retun if import sucessful'
python test/test_torch_geometric.py
