FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PYTHONPATH=third_party/Matcha-TTS

# Combine apt installs + cleanup in ONE layer to keep image slim
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev build-essential \
        sox libsox-dev \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Cython + numpy via pip first (needed for pyworld compilation)
RUN pip install cython numpy --no-cache-dir \
        -i https://mirrors.aliyun.com/pypi/simple/ \
        --trusted-host=mirrors.aliyun.com

RUN pip install pyworld --no-cache-dir \
        -i https://mirrors.aliyun.com/pypi/simple/ \
        --trusted-host=mirrors.aliyun.com


# Use aliyun mirror for pip
RUN pip install pynini==2.1.5 --no-cache-dir \
        -i https://mirrors.aliyun.com/pypi/simple/ \
        --trusted-host=mirrors.aliyun.com

WORKDIR /app

# Install Python deps (layer cached as long as requirements.txt doesn't change)
COPY requirements.txt /app
RUN pip install -r requirements.txt --no-cache-dir \
        -i https://mirrors.aliyun.com/pypi/simple/ \
        --trusted-host=mirrors.aliyun.com \
    && rm -rf ~/.cache/pip

# Copy pretrained models and remove ModelScope temp directories
COPY iic /app/pretrained_models
RUN rm -rf /app/pretrained_models/*/\._____temp/

# Install ttsfrd from local wheels + extract resource
RUN cd pretrained_models/CosyVoice-ttsfrd/ \
    && pip install ttsfrd_dependency-0.1-py3-none-any.whl --no-cache-dir \
    && pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl --no-cache-dir \
    && unzip -o resource.zip -d . \
    && rm -rf ~/.cache/pip

# Copy source code
COPY cosyvoice /app/cosyvoice
COPY third_party /app/third_party
COPY main.py /app

# TLS certs (optional — mount as volume in production instead)
COPY cert.pem /app
COPY key.pem /app

CMD ["python3", "main.py"]
