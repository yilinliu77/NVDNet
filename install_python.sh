ROOT_DIR=$PWD

# Miniconda3
# wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -O miniconda.sh && bash miniconda.sh -b -p /root/miniconda3 && rm miniconda.sh
# apt install libgl1-mesa-glx libegl1-mesa-dev

conda config --set auto_update_conda false
# conda install -c intel mkl mkl-devel mkl-static mkl-include -y
conda install -c pytorch -c nvidia pytorch=2.0.1 torchvision=0.15.2 torchaudio=2.0.2 pytorch-cuda=11.8 -y
# pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
# conda install -c pyg pyg  -y
# pip install torch_geometric 
# pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install PyMCubes pytorch-lightning hydra-core shapely scikit-image matplotlib tensorboard plyfile opencv-python opencv-contrib-python open3d ray[default] h5py
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
