FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
LABEL Name=cv Version=0.0.1

RUN apt update -y
RUN apt install -y git
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r /workspace/requirements.txt -U --extra-index-url https://download.pytorch.org/whl/cu113
COPY . /workspace/cv/
WORKDIR /workspace/cv/
RUN rm -rf /workspace/requirements.txt
RUN apt-get -y update && apt-get install tmux -y
RUN apt-get install ffmpeg libsm6 libxext6  -y