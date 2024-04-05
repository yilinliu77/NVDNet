FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

LABEL maintainer="whatsevenlyl@gmail.com"

SHELL ["/bin/bash", "--login", "-c"]

# #######################################################################################
# Read ME
# Set the proxy host. Change the IP and port in line 12-14
# #######################################################################################
# Proxy, commet line 11~13 if no proxy has been used
ARG USE_PROXY true
ARG PROXY_IP=192.168.1.101
ARG PROXY_PORT=7890
ARG PROXY=http://$PROXY_IP:$PROXY_PORT
ARG DISPLAY=$PROXY_IP:0.0
# #######################################################################################
# #######################################################################################
ENV HTTP_PROXY=$PROXY
ENV HTTPS_PROXY=$PROXY
# RUN apt update && apt install wget -y
RUN \
    # echo "nameserver 114.114.114.114" > /etc/resolv.conf && \
    rm /etc/apt/sources.list.d/cuda.list && \
    sed -i "s/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g" /etc/apt/sources.list && \
    sed -i "s/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g" /etc/apt/sources.list && \
    apt update && \
    apt install wget -y

# apt dependencis
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && \
    apt install -y swig libblas-dev liblapack-dev sudo tmux git cmake libx11-dev libxft-dev bison gperf libglfw3-dev libxi-dev libxcursor-dev libxdamage-dev libxinerama-dev libxinerama-dev libxcursor-dev xorg-dev libglu1-mesa-dev gcc-9 gdb make gdb git cmake wget curl zip unzip tar python3 sudo libtool autoconf python3-distutils autoconf-archive net-tools proxychains4
RUN sed -i "s/socks.*/http $PROXY_IP $PROXY_PORT/g" /etc/proxychains4.conf && sed -i "s/#quiet.*/quiet_mode/g" /etc/proxychains4.conf &&\
    echo 'set completion-ignore-case On' >> ~/.inputrc && \
    echo "root:root" | chpasswd && \
    apt update && \
    apt install -y openssh-server vim && \
    echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    service ssh restart
RUN sh -c "$(wget -e use_proxy=yes -e https_proxy=$PROXY -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.4/zsh-in-docker.sh)" -- \
    -p git \
    -p https://github.com/agkozak/zsh-z \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions \
    && \
    chsh -s $(which zsh) root
    
# ======================================================================================================
#                                           Install Python library
# ======================================================================================================

RUN wget -e use_proxy=yes -e https_proxy=$PROXY --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    /opt/conda/condabin/conda init zsh
    # echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.zshrc && \
    # echo "conda activate base" >> ~/.zshrc

RUN echo "export LC_ALL=en_US.UTF-8" >> ~/.zshrc && \
    echo "export LANG=en_US.UTF-8" >> ~/.zshrc && \
    echo "export PATH=/usr/local/cuda/bin/:\${PATH}"  >> ~/.zshrc && \
    echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> ~/.zshrc && \
    echo "alias p=proxychains4"  >> ~/.zshrc && \
    echo "alias pp='export HTTP_PROXY=$PROXY && export HTTPS_PROXY=$PROXY'"  >> ~/.zshrc && \
    echo "unsetopt nomatch"  >> ~/.zshrc && \
    echo "DISPLAY=$DISPLAY"  >> ~/.zshrc

# ======================================================================================================
#                                           Other stuff
# ======================================================================================================
EXPOSE 22 6100 6101 6102 15000 15001
CMD ["/usr/sbin/sshd", "-D"]
