FROM python:latest
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install torch_geometric
RUN pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
COPY requirements.txt /home/gns/requirements.txt
RUN pip3 install -r requirements.txt
WORKDIR /home/gns
RUN /bin/bash