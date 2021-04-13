FROM pytorch/pytorch
LABEL Name=cv Version=0.0.1

RUN apt update -y
RUN apt install -y git
COPY requirements.txt /workspace/requirements.txt
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt
RUN rm -rf /workspace/requirements.txt
RUN apt-get -y update && apt-get install tmux -y