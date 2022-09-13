# syntax=docker/dockerfile:experimental
FROM --platform=linux/x86_64 nvidia/cuda:11.6.0-runtime-ubuntu20.04

#FROM ubuntu:18.04

ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    git \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    pkg-config

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
RUN conda create -y -n torchbeast python=3.9 \
    protobuf \
    ninja \
    pyyaml \
    mkl \
    mkl-include \
    setuptools \
    cmake \
    cffi \
    typing

# Activate environment in .bashrc.
RUN echo "conda activate torchbeast" >> /root/.bashrc

# Make bash excecute .bashrc even when running non-interactively.
ENV BASH_ENV /root/.bashrc

ADD . /src
RUN pip install -r /src/requirements.txt



ENV OMP_NUM_THREADS 1

# Run.
CMD ["bash", "-c", "train.sh"]

# Docker commands:
#   docker rm torchbeast -v
#   docker build -t torchbeast .
#   docker run --name torchbeast torchbeast
# or
#   docker run --name torchbeast -it torchbeast /bin/bash
