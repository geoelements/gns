#!/bin/bash

module reset 

# create env
# ---------
source module.sh

python3 -m virtualenv venv
source venv/bin/activate

which python
python -m pip install --upgrade pip
pip3 install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
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
