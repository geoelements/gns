FROM python:3.11

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install torch_geometric && \
    pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
RUN pip3 install -r requirements.txt

ENV PYTHONPATH=/app

# Add Python path to PATH
ENV PATH="/usr/local/bin:${PATH}"

# Create a bash script to set up the environment
RUN echo '#!/bin/bash\n\
export PYTHONPATH=/app\n\
export PATH="/usr/local/bin:$PATH"\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]