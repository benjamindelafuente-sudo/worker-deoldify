# worker-deoldify-VIDEO — RunPod Serverless worker para colorizar video.
# Estrategia: usar la imagen oficial del repo kodxana/worker-deoldify
# (que ya viene con DeOldify Artistic bakeado para imagenes) y agregar
# soporte para video en runtime via loop de frames.
#
# Esto evita reconstruir DeOldify+wandb+fastai desde cero (que tiene
# conflictos de dependencias en torch 2.x).

FROM ghcr.io/kodxana/worker-deoldify:latest

# Nos aseguramos de que el modelo Video (no Artistic) este presente.
# La imagen base viene con Artistic, asi que descargamos el Video al lado.
USER root

# Verificar ffmpeg presente
RUN which ffmpeg || apt-get update -y && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# Bajar el modelo Video (~285 MB) — distinto del Artistic que ya viene.
# El modelo Video es mas estable para video (menos flickr).
RUN mkdir -p /DeOldify/models && \
    cd /DeOldify/models && \
    wget -q https://data.deepai.org/deoldify/ColorizeVideo_gen.pth -O ColorizeVideo_gen.pth && \
    wget -q https://data.deepai.org/deoldify/ColorizeVideo_crit.pth -O ColorizeVideo_crit.pth && \
    ls -la

# Reescribir el handler.py con nuestra version de video
COPY src/handler.py /app/handler.py
COPY src/rp_schemas.py /app/rp_schemas.py

WORKDIR /app

# runpod serverless entrypoint
CMD ["python3", "-u", "/app/handler.py"]
