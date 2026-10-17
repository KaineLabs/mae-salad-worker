# Grandma Mae's Salad worker: ComfyUI (generation) + ostris ai-toolkit (LoRA training), models downloaded at start.
# Built by GitHub Actions (.github/workflows/build.yml) and pushed to ghcr.io. CUDA 13 wheels: host driver >= 580.
FROM nvidia/cuda:13.0.3-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 HF_XET_HIGH_PERFORMANCE=1 HF_HUB_ENABLE_HF_TRANSFER=0
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-dev git ffmpeg build-essential \
      ca-certificates curl libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
ARG TORCH="torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0"
# ComfyUI at the commit tested on RunPod (LTX-2.3 ia2v, LTX-2.5, Qwen-Image-Edit-2511 int8, Z-Image)
RUN python3 -m venv /opt/comfy-venv && git clone -q https://github.com/comfyanonymous/ComfyUI /opt/ComfyUI \
 && cd /opt/ComfyUI && git checkout -q 06c6780 \
 && /opt/comfy-venv/bin/pip install $TORCH --index-url https://download.pytorch.org/whl/cu130 \
 && /opt/comfy-venv/bin/pip install -r requirements.txt huggingface_hub boto3 pyyaml
# ai-toolkit pinned (Z-Image Turbo LoRA training with the de-distill adapter)
RUN python3 -m venv /opt/aitk-venv && git clone -q https://github.com/ostris/ai-toolkit /opt/ai-toolkit \
 && cd /opt/ai-toolkit && git checkout -q 577004a9cd281035f3913f0db7ef34616e04b6e1 && git submodule update --init --recursive -q \
 && /opt/aitk-venv/bin/pip install $TORCH --index-url https://download.pytorch.org/whl/cu130 \
 && /opt/aitk-venv/bin/pip install -r requirements.txt && /opt/aitk-venv/bin/pip install setuptools==69.5.1
COPY . /opt/worker
CMD ["/opt/comfy-venv/bin/python", "/opt/worker/main.py"]
