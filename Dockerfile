FROM condaforge/miniforge3:latest
RUN conda install --yes dascore && conda clean --all --yes
