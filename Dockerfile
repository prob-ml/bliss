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
RUN mkdir ./bliss && touch ./bliss/__init__.py
RUN mkdir ./case_studies && touch ./case_studies/__init__.py
RUN poetry install --no-interaction --ansi
COPY bliss ./bliss/
COPY case_studies ./case_studies/
COPY tests ./tests/
COPY data ./data/
COPY typings ./typings/
COPY .darglint ./
COPY .flake8 ./
COPY .pylintrc ./
RUN find -name "*.pyc" -exec rm {} \;
CMD ["bash"]
