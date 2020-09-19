FROM nvidia/cuda:10.1-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y
RUN apt-get install libsm6 -y \
    libxext6 -y \
    libxrender-dev -y \
    python3-pip -y \
    tzdata -y 

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install psutil

COPY . /home/app/servidor
WORKDIR /home/app/servidor

