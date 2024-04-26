ROOT_DIR=$PWD

# Miniconda3
# wget https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh -O miniconda.sh && bash miniconda.sh -b -p /root/miniconda3 && rm miniconda.sh
# apt install libgl1-mesa-glx libegl1-mesa-dev

conda config --set auto_update_conda false
conda install -c pytorch -c nvidia -c pyg -c conda-forge pythonocc-core=7.7 pytorch=2.0.1 torchvision=0.15.2 torchaudio=2.0.2 pytorch-cuda=11.8 pyg --solver=libmamba -y
pip install PyMCubes pytorch-lightning hydra-core shapely scikit-image matplotlib tensorboard plyfile opencv-python opencv-contrib-python open3d ray[default] h5py trimesh igraph
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
