FROM ubuntu
MAINTAINER jota.junior@beblue.com.br

RUN apt-get update && \
	apt-get install -y python3 \
	libpq-dev \
	python3-pip  \
	python3-numpy \
	python3-pandas \
	gfortran \
	libblas-dev \
	liblapack-dev \
	libatlas-base-dev \
	python3-dev \
	git

COPY . /wallethub
WORKDIR /wallethub

RUN pip3 install \
	git+git://github.com/jotajunior/fancyimpute.git \
	scikit-learn==0.18.1 \
	pandas \
	numpy

RUN rm -rf /var/lib/apt/lists/*

CMD /bin/bash
