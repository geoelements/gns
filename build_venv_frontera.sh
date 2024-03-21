#!/bin/bash

module reset 

# Load modules
module load python3/3.9
module load cuda/12

# create env
# ---------
source module.sh

python3 -m virtualenv env
source env/bin/activate

which python
python -m pip install --upgrade pip
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url 
https://download.pytorch.org/whl/cu118
pip install torch_geometric==2.4.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster 
torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install -r requirements.txt

# test env
# --------

echo 'which python -> env'
which python

echo 'test_pytorch.py -> random tensor'
python3 test/test_pytorch.py 

echo 'test_pytorch_cuda_gpu.py -> True if GPU'
python3 test/test_pytorch_cuda_gpu.py

echo 'test_torch_geometric.py -> no return if import successful'
python3 test/test_torch_geometric.py

echo 'waterdrop sample/Dataset -> no return if import successful'
python3 -m gns.train --data_path="./gns-sample/WaterDropSample/dataset/" --model_path="../models/" --output_path="../output/"  -ntraining_steps=100

# Clean up
# --------
#deactivate
#rm -r env

