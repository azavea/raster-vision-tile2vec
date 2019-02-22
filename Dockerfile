FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04
ARG PYTHON_VERSION=3.6

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libjpeg-dev \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH
RUN conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include cython typing
RUN conda install -y -c pytorch magma-cuda92 torchvision
RUN conda install -y -c fastai fastai
RUN conda clean -ya

# Install all tile2vec dependencies
COPY tile2vec-environment.yml /opt/config/environment.yml
RUN conda install -y -c conda-forge gdal libiconv libgdal curl libtiff libkml
RUN conda env update -f=/opt/config/environment.yml

RUN pip install nvidia-ml-py3 dataclasses imageio sklearn

# Install Jupyter
RUN pip install jupyter matplotlib
RUN pip install jupyter_http_over_ws
RUN jupyter serverextension enable --py jupyter_http_over_ws

### Install rastervision
RUN mkdir /opt/tmp
WORKDIR /opt/tmp

RUN pip install boto3==1.7.* awscli==1.15.*
RUN git clone https://github.com/lossyrob/raster-vision && \
    cd raster-vision && \
    git checkout rde/feature/parallelize-tile2vec && \
    pip install -r requirements.txt && \
    python setup.py install

RUN rm -r /opt/tmp
WORKDIR /opt/src

### rastervision

### Install Faiss
RUN conda install faiss-gpu cuda92 -c pytorch

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Certs for rasterio
RUN mkdir -p /etc/pki/tls/certs
RUN cp /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt

COPY tile2vec /opt/src/tile2vec

ENV PYTHONPATH=/opt/src:$PYTHONPATH
