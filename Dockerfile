FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update -y && \
    apt-get install -y ffmpeg libgl1 libglib2.0-0 wget git && \
    rm -rf /var/lib/apt/lists/*

# Python deps — DeOldify + a wget helper. Pinned for reproducibility.
RUN pip install --no-cache-dir \
    "torch==2.4.0" \
    "torchvision==0.19.0" \
    "fastai==2.7.18" \
    "fastcore==1.5.29" \
    "wandb<0.16" \
    "fsspec<2024.10" \
    "numpy<2" \
    "opencv-python-headless" \
    "Pillow" \
    "requests"

# Clone DeOldify (jantic/DeOldify, MIT). Pin a commit estable.
ARG DEOLDIFY_COMMIT=523fd34
RUN git clone https://github.com/jantic/DeOldify.git /DeOldify && \
    cd /DeOldify && \
    git checkout ${DEOLDIFY_COMMIT}

# Patch: el repo original usa wandb>=0.16 que rompe el import. Downgradeamos via pip arriba.
# Patch: silence warnings, install any missing subdeps
RUN pip install --no-cache-dir -r /DeOldify/requirements.txt || true

# Bake the Video colorizer weights (~285 MB) — optimized for flicker-free video.
# data.deepai.org is the official mirror (same model as in the original repo).
RUN mkdir -p /DeOldify/models && \
    cd /DeOldify/models && \
    wget -q https://data.deepai.org/deoldify/ColorizeVideo_gen.pth && \
    wget -q https://data.deepai.org/deoldify/ColorizeVideo_crit.pth

# Pre-download any other resources DeOldify needs on first run
ENV DEOLDIFY_DEVICE=cuda

WORKDIR /app

# Copy our serverless handler
COPY handler.py /app/handler.py
COPY rp_schemas.py /app/rp_schemas.py

# DeOldify modules are at /DeOldify — add to path
ENV PYTHONPATH=/DeOldify:/app

# RunPod serverless entrypoint
CMD ["python3", "-u", "/app/handler.py"]