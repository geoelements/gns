# Graph Network Simulator

## Getting Started

### Building Environment on Frontera

- ssh to frontera, start an idev session on rtx node (i.e., GPU-enabled node)
- run the follow to setup a virtualenv

```
ml use /work2/02604/ajs2987/frontera/apps/modulefiles
ml cuda/11.1
module load python3/3.9.2

python3 -m virtualenv venv
source venv/bin/activate

which pip3
pip3 install torch==1.10.0+cu111 torchvision==0.11.1+cu111 torchaudio==0.10.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip3 install torch-spline-conv -f https://data.pyg.org/whl/torch-1.10.0+cu111.html --no-binary torch-spline-conv
pip3 install torch-scatter torch-sparse torch-cluster torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
```

- test gpu install by running script

```python
import torch
print(torch.cuda.is_available()) # --> True
```
