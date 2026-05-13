FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda

RUN #sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get update -y
RUN apt-get -y install git unzip git-lfs g++ python3 python3-pip
RUN git lfs install

WORKDIR /opt/CosyVoice

COPY requirements.txt .
# here we use python==3.10 because we cannot find an image which have both python3.8 and torch2.0.1-cu118 installed
RUN pip install openai-whisper==20231117  -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com --no-cache-dir  --no-build-isolation
RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com --no-cache-dir

COPY pretrained_models/CosyVoice-ttsfrd/ pretrained_models/CosyVoice-ttsfrd
RUN cd  pretrained_models/CosyVoice-ttsfrd && tar -zxvf resource.tar && pip install ttsfrd_dependency-0.1-py3-none-any.whl && pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
COPY asset  asset
COPY app.py app.py
COPY cosyvoice cosyvoice


CMD [ "python3", "app.py", "--model_dir=pretrained_models/CosyVoice2-0.5B" ]
