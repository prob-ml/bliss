FROM docker.io/nvidia/cuda:11.0.3-base-ubuntu20.04
RUN apt update
RUN apt install -y \
    python3 \
    python3-pip \
    python-is-python3 \
    libfftw3-dev \
    git-lfs \
    wget \
    curl \
    vim
RUN python -m pip install poetry
ENV BLISS_HOME=/workspaces/bliss
WORKDIR /workspaces/bliss
COPY pyproject.toml poetry.lock ./
COPY bliss ./bliss/
COPY case_studies ./case_studies/
COPY tests ./tests/
COPY data ./data/
COPY typings ./typings/
RUN find -name "*.pyc" -exec rm {} \;
RUN poetry install --no-interaction --ansi
CMD ["bash"]
