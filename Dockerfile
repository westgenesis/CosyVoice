FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive


RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get update -y
RUN apt-get -y install git unzip git-lfs g++
RUN git lfs install

WORKDIR /opt/CosyVoice

COPY requirements.txt .
# here we use python==3.10 because we cannot find an image which have both python3.8 and torch2.0.1-cu118 installed
RUN pip install openai-whisper==20231117  -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com --no-cache-dir  --no-build-isolation
RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com --no-cache-dir

COPY third_party third_party
# 替换third_party里面的requirements.txt 的diffusers==0.25.0为diffusers==0.29.0
RUN sed -i 's/diffusers==0.25.0/diffusers==0.29.0/g' third_party/Matcha-TTS/requirements.txt
RUN pip install -e third_party/Matcha-TTS -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com --no-cache-dir
COPY asset  asset
COPY app.py app.py
COPY cosyvoice cosyvoice


CMD [ "python3", "app.py", "--model_dir=pretrained_models/CosyVoice2-0.5B" ]