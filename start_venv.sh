#!/bin/bash

module reset

# start env
# ---------
ml use /work2/02064/ajs2987/frontera/apps/modulefiles
ml cuda/11.1
ml nccl
ml cudnn

module load phdf5
module load python3/3.9

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
