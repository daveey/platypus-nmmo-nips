# syntax=docker/dockerfile:experimental
FROM nvidia/cuda:11.6.0-runtime-ubuntu20.04

#FROM ubuntu:18.04

ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

RUN set -e; \
    apt-get update -y && apt-get install -y \
    git \
    vim \
    cmake \
    curl \
    g++ \
    git \
    less \
    wget \
    tini \
    lsb-release; \
    GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`; \
    echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | \
    tee /etc/apt/sources.list.d/gcsfuse.list; \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    apt-key add -; \
    apt-get update; \
    apt-get install -y gcsfuse \
    && apt-get clean

WORKDIR /src/baseline

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

RUN bash Miniconda3-latest-Linux-x86_64.sh -b

ENV PATH /root/miniconda3/bin:$PATH

ENV CONDA_PREFIX /root/miniconda3/envs/torchbeast

# Clear .bashrc (it refuses to run non-interactively otherwise).
RUN echo > ~/.bashrc

# Add conda logic to .bashrc.
RUN conda init bash

# Create new environment and install some dependencies.
RUN conda create -y -n torchbeast python=3.9

# Activate environment in .bashrc.
RUN echo "conda activate torchbeast" >> /root/.bashrc

# Make bash excecute .bashrc even when running non-interactively.
ENV BASH_ENV /root/.bashrc

ADD . /src
RUN pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html 
RUN pip install -r /src/requirements.txt



ENV OMP_NUM_THREADS 1

# Run.
CMD ["bash", "-c", "./train.sh"]

# Docker commands:
#   docker rm torchbeast -v
#   docker build -t torchbeast .
#   docker run --name torchbeast torchbeast
# or
#   docker run --name torchbeast -it torchbeast /bin/bash
