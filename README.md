# Graph Network Simulator

## Getting Started

### Building Environment on Frontera

- ssh to frontera, start an idev session on rtx node (i.e., GPU-enabled node)
- download miniconda: `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
- install miniconda: `bash Miniconda3-latest-Linux-x86_64.sh`
- after install start a new shell: `bash`
- create new conda env: `conda create --name gnn python=3.8`
- activate new env: `conda activate gnn`
- load system modules: `module load cuda cudnn`
- install pytorch: `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`
- install pyg: `conda install pyg -c pyg -c conda-forge`

- test gpu install by running script

```python
import torch
print(torch.cuda.is_available()) # --> True
```
