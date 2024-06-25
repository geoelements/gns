FROM python:latest

WORKDIR /home/gns

COPY requirements.txt .

RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install torch_geometric && \
    pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html && \
    pip3 install -r requirements.txt

CMD ["/bin/bash"]