#!/bin/bash

module reset # -> default is python3/3.7.0 (frontera)
             # -> default is python3/3.9.7 (ls6)


# background
# ----------

# pytorch
# pytorch 1.10 + cuda -> requires cuda 10.2 or 11.3
# pytorch 1.9 + cuda -> (not listed for some reason?)
#                       based on pygeometric it supports cuda 10.2 or 11.1
# pytorch 1.8 + cuda -> requires cuda 10.2 or 11.1

# pygeometric 
# pygeometric -> requires pytorch >= 1.8 

# frontera
# only has cuda 10 10.1 11.0 11.3 

# ls6
# only has cuda 11.4

# create env
# ---------
ml use /work2/02064/ajs2987/frontera/apps/modulefiles
ml cuda/11.1
ml cudnn
ml nccl

module load phdf5
module load python3/3.9

python3 -m virtualenv venv
source venv/bin/activate

which pip3
pip3 install torch==1.10.0+cu111 torchvision==0.11.1+cu111 torchaudio==0.10.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip3 install torch-spline-conv -f https://data.pyg.org/whl/torch-1.10.0+cu111.html --no-binary torch-spline-conv
pip3 install torch-scatter torch-sparse torch-cluster torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
pip3 install -r requirements.txt

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

