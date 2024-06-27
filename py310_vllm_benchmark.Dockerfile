# Use an official NVIDIA runtime as a parent image
FROM nvcr.io/nvidia/pytorch:23.10-py3
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-10.html#rel-23-10
# Set the working directory in the container
USER root
ARG user=inference_ai
RUN apt-get update && apt-get install -y sudo
RUN useradd --create-home --no-log-init --shell /bin/bash ${user}
    #&& groupadd sudo \
RUN usermod -aG sudo ${user}
RUN echo "${user}:1" | chpasswd
RUN usermod -u 1000 ${user} && usermod -G 1000 ${user}

## Install Python3 
## TimeZone Settings
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles
RUN apt-get install -y tzdata
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
#RUN apt-get update
RUN apt-get install -y --no-install-recommends\
    git \
    wget \
    curl \
    build-essential \
    libffi-dev \
    libgdbm-dev \
    libc6-dev \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    liblzma-dev \
    vim

RUN git clone https://github.com/vllm-project/vllm.git && cd vllm && pip install -e .
RUN cd vllm/benchmarks && wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json -O sharegpt_dataset.json 
# Make port 8888 available to the world outside this container
EXPOSE 8888
EXPOSE 80
EXPOSE 22
# Make 8073 port available for the demos
EXPOSE 8073
EXPOSE 8051