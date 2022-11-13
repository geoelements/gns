FROM continuumio/anaconda3:latest
RUN conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cpuonly -c pytorch
RUN conda install pyg -c pyg
RUN conda install -c anaconda absl-py 
RUN conda install -c conda-forge numpy
RUN conda install -c conda-forge dm-tree
RUN conda install -c conda-forge matplotlib-base
RUN conda install -c conda-forge pyevtk
WORKDIR /home/gns
RUN /bin/bash