FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG USER=vscode
RUN apt update \
    && apt install -y --no-install-recommends curl wget git zsh sudo build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev htop ffmpeg libsm6 libxext6 \
    && apt autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && usermod -s /usr/bin/zsh ${USER}

USER ${USER}
ARG HOME="/home/${USER}"
WORKDIR ${HOME}

ARG PYTHON_VERSION=3.9
ENV PYENV_ROOT=${HOME}/.pyenv
ARG PYENV_PATH="${PYENV_ROOT}/bin:${PYENV_ROOT}/shims"
ENV PATH="${PYENV_PATH}:$PATH"
RUN set -x \
    && curl http://pyenv.run | bash \
    && echo 'eval "$(pyenv init -)"' >>${HOME}/.zshrc \
    && pyenv install -v ${PYTHON_VERSION} \
    && pyenv global ${PYTHON_VERSION}

RUN pip install face-recognition \
    && pip install opencv-python