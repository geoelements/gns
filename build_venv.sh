#!/bin/bash

module reset 

# create env
# ---------
ml cuda/11.3
ml cudnn
ml nccl

module load phdf5
module load python3/3.9
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

python3 -m virtualenv venv
source venv/bin/activate

which python
python -m pip install --upgrade pip
python -m pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.11.0+cu113.html --no-binary torch-spline-conv
python -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
python -m pip install -r requirements.txt

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

# Clean up
# --------
#deactivate
#rm -r venv
