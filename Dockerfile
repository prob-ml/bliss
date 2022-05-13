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
RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser -m appuser 
USER appuser
WORKDIR /workspaces/bliss
COPY pyproject.toml poetry.lock ./
COPY bliss ./bliss/
COPY case_studies ./case_studies/
COPY tests ./tests/
RUN poetry install --no-interaction --ansi
USER root
RUN chown appuser /workspaces/bliss
RUN find | grep .pyc$ | xargs rm
COPY data ./data/
COPY typings ./typings/
USER appuser
ENV BLISS_HOME=/workspaces/bliss
CMD ["bash"]
