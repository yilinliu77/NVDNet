FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
LABEL maintainer="whatsevenlyl@gmail.com"

SHELL ["/bin/bash", "--login", "-c"]

# RUN apt update && apt install wget -y
RUN apt update && \
    apt install wget -y

# apt dependencis
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && \
    apt install -y swig libblas-dev liblapack-dev sudo tmux git cmake libx11-dev libxft-dev bison gperf libglfw3-dev libxi-dev libxcursor-dev libxdamage-dev libxinerama-dev libxinerama-dev libxcursor-dev xorg-dev libglu1-mesa-dev gcc-9 gdb make gdb git cmake wget curl zip unzip tar python3 sudo libtool autoconf python3-distutils autoconf-archive net-tools proxychains4 libdbus-1-dev
RUN echo 'set completion-ignore-case On' >> ~/.inputrc && \
    echo "root:root" | chpasswd && \
    apt update && \
    apt install -y openssh-server vim && \
    echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    service ssh restart
RUN sh -c "$(wget -e -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.4/zsh-in-docker.sh)" -- \
    -p git \
    -p https://github.com/agkozak/zsh-z \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions \
    && \
    chsh -s $(which zsh) root

RUN echo "export LC_ALL=en_US.UTF-8" >> ~/.zshrc && \
    echo "export LANG=en_US.UTF-8" >> ~/.zshrc && \
    echo "export PATH=/usr/local/cuda/bin/:\${PATH}"  >> ~/.zshrc && \
    echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> ~/.zshrc && \
    echo "unsetopt nomatch"  >> ~/.zshrc 

RUN cd /root/ && \
    git clone --recursive https://github.com/yilinliu77/NVDNet && \
    cd NVDNet && chmod +x install.sh && ./install.sh

EXPOSE 22 6100 6101 6102 15000 15001
CMD ["/usr/sbin/sshd", "-D"]