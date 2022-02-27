FROM nvidia/cudagl:11.1-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y git python3 python3-pip vim cmake ffmpeg

# Make image smaller by not caching downloaded pip pkgs
ARG PIP_NO_CACHE_DIR=1

# Install pytorch for example, and ensure sim works with all our required pkgs
ARG TORCH=1.10.0
ARG CUDA=cu111
# Pytorch and torch_geometric w/ deps
RUN pip3 install torch==${TORCH}+${CUDA} \
    -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html \
    torch_geometric
# pytorch_geometric can be a bit buggy during install
RUN python3 -c "import torch; import torch_geometric"

ADD ./src/rllib_multi_agent_demo/requirements.txt \
    /build/requirements/requirements_demo.txt
ADD ./requirements.txt \
    /build/requirements/requirements_app.txt

RUN pip3 install \
    -r /build/requirements/requirements_demo.txt \
    -r /build/requirements/requirements_app.txt

# Make PyGame render in headless mode
ENV SDL_VIDEODRIVER dummy
ENV SDL_AUDIODRIVER dsp

WORKDIR /home
