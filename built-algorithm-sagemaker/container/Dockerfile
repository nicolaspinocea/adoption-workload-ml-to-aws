FROM ubuntu:20.04

MAINTAINER Amazon AI <sage-learner@amazon.com>


RUN apt-get -y update && apt-get install -y --no-install-recommends \
         libgomp1 \
         wget \
         python3-pip \
         python3-setuptools \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
#RUN ln -s /usr/bin/pip3 /usr/bin/pip

RUN pip --no-cache-dir install numpy scikit-learn pandas flask gunicorn lightgbm==3.3.5

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

# Set up the program in the image
COPY lightgbm /opt/program
WORKDIR /opt/program
