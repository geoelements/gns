#!/bin/bash

module reset 

# Load modules
module load python3/3.9
module load cuda/12

# create env
# ---------
python3 -m virtualenv venv
source venv/bin/activate

python -m pip install --upgrade pip
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric==2.4.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster 
torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install -r requirements.txt

# test env
# --------

# Check if the first command line argument is "--run-tests=true"
if [ "$1" = "--run-tests=true" ]; then
  echo 'Running tests...'
  
  echo 'which python -> venv'
  which python
  
  echo 'test_pytorch.py -> random tensor'
  python3 test/test_pytorch.py
  
  echo 'test_pytorch_cuda_gpu.py -> True if GPU'
  python3 test/test_pytorch_cuda_gpu.py
  
  echo 'test_torch_geometric.py -> no return if import successful'
  python3 test/test_torch_geometric.py
  
else
  echo "Skipping tests. To run tests, use the argument --run-tests=true"
fi

# Clean up
# --------
#rm -r venv
