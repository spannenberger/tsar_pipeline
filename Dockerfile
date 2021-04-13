FROM pytorch/pytorch
LABEL Name=cv Version=0.0.1

COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt
RUN rm -rf /workspace/requirements.txt
RUN apt-get -y update && apt-get install tmux -y