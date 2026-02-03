FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt update && apt install -y \
    python3 \
    python3-pip \
    git \
    tmux \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

CMD ["bash"]
