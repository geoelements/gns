FROM continuumio/anaconda3:latest
RUN conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cpuonly -c pytorch
RUN conda install pyg -c pyg
WORKDIR /home/gns
RUN /bin/bash