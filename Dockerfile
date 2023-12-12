# syntax=docker/dockerfile:1

FROM ubuntu:22.04
RUN apt update && apt install -y wget
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN chmod +x Miniconda3-latest-Linux-x86_64.sh
RUN ./Miniconda3-latest-Linux-x86_64.sh  -b -p /opt/miniconda3
RUN apt autoremove && apt autoclean
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/opt/miniconda3/bin:$PATH"
RUN conda install dascore -c conda-forge
