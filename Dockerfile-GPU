FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
RUN apt-get update
RUN apt-get upgrade -y

RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y git

RUN pip install --upgrade pip ipython ipykernel

COPY requirements.txt requirements.txt
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu118
RUN pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
RUN pip install torch_geometric
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
RUN pip install absl-py autopep8 numpy==1.23.1 dm-tree matplotlib pyevtk pytest tqdm toml
RUN pip install -r requirements.txt



CMD ["/bin/bash"]