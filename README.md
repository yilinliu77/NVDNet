# NVDNet

[[Paper](https://arxiv.org/abs/2406.05261)] [[Video](https://youtu.be/omS38sAbt9w)]

Official implementation of [Split-and-Fit: Learning B-Reps via Structure-Aware Voronoi Partitioning](https://arxiv.org/abs/2406.05261) from [Yilin Liu](https://yilinliu77.github.io/), Jiale Chen, Shanshan Pan, [Daniel Cohen-Or](https://danielcohenor.com/), [Hao Zhang](https://www.cs.sfu.ca/~haoz/) and [Hui Huang](https://vcc.tech/~huihuang).

![](assets/1teaser.png)

# Citation
```
@article {liu_sig24,
  author = {Yilin Liu and Jiale Chen and Shanshan Pan and Daniel Cohen-Or and Hao Zhang and Hui Huang},
  title = {{Split-and-Fit}: Learning {B-Reps} via Structure-Aware {Voronoi} Partitioning},
  journal = {ACM Transactions on Graphics (Proc. of SIGGRAPH)},
  volume = {43},
  number = {4},
  pages = {108:1--108:13},
  year = {2024}
}
```

# Install

We provide three options to install NVDNet: packed docker container, Dockerfile, and mannul. We recommend using packed docker container for the ease of use.

## Option 1: Packed Docker Container (Recommended)

Download the packed docker container including pre-compiled environment and sample data from [here](https://www.dropbox.com/scl/fi/p5zi97igmiygv2x69row7/NVD.tar?rlkey=l5o7hx5qu1zymxd7k4wln1pfq&st=o6zuw47v&dl=0) (~30GB) and load it using the following command:

```
cat NVD.tar | docker import - nvd_release:latest
```

Run the container using the following command:

```
docker run -d -p 6100:22 --name nvdnet -v /path/to/your/data:/data nvd_release:latest
```

Then you can ssh into the container using the following command:

```
ssh -p 6100 root@127.0.0.1 (password: root)
```

## Option 2: Dockerfile

```bash
cd docker && docker build -t nvdnet .

docker run -it -d -p 6100:22 --name nvdnet -v /path/to/your/data:/data nvdnet

ssh -p 6100 root@127.0.0.1 (password: root)
```

## Option 3: Manual

Please refer to `Dockerfile`, `install.sh` and `install_python` for manual installation. This script has been tested on Ubuntu 22.04 with cuda 11.8.0.

# Usage

## Inference (Single point cloud)

Download the weights from [here](https://www.dropbox.com/scl/fi/g6k2zwt2bbkk9y3vycrmo/0102_v15-_parsenet_unet_base16_focal75_wonormal_channel4_float32_aug.ckpt?rlkey=smryoqn52woonwhreo6l3oe1b&st=dbjvpsib&dl=0), the packed docker container has already included the weights.

## Prepare data

Put your point cloud data (w/ or w/o normal) under `/data/poisson`. We provide two sample point clouds in that folder.

The input folder should be organized as follows:

```
- root
| - poisson
| | - sample.ply (your point cloud here)
```

Replace the `{data root}` with your data path (`/data/` in this case):

```
export DATA_ROOT={data root}
export ROOT_PATH=/root/NVDNet
```

### Generate UDF field using NDC

This step will generate the UDF field for the input point cloud using [NDC](https://github.com/czq142857/NDC). The output will be saved in `${DATA_ROOT}/feat/ndc`. 

Follow the instruction of NDC if you are not using the packed container. You can speed up the process if you have multiple GPUs by replacing `1` with the number of GPUs you have. You can also use `CUDA_VISIBLE_DEVICES=x` to restrict the GPU usage.

`cd ${ROOT_PATH} && python generate_test_voronoi_data.py ${DATA_ROOT} 1`

`${ROOT_PATH}/build/prepare_evaluate_gt_feature/prepare_evaluate_gt_feature ${DATA_ROOT}/ndc_mesh ${DATA_ROOT}/feat/ndc --resolution 256`


### Voronoi prediction

This step will use NVDNet to predict the Voronoi diagram for the UDF field. The output Voronoi and the visualization will be saved in `${DATA_ROOT}/voronoi`. 

`cd ${ROOT_PATH}/python && python test_model.py dataset.root={DATA_ROOT} dataset.output_dir={DATA_ROOT}/voronoi`

### Primitive extraction

This step will extract the mesh from the Voronoi diagram. The output mesh will be saved in `${DATA_ROOT}/mesh`.

`${ROOT_PATH}/build/extract_mesh/extract_mesh `

# Evaluation on test set

Download the test data from [here](https://www.dropbox.com/scl/fi/sxhyd2x25tcpe8tp9cmko/NVD_test_data.zip?rlkey=zew313ul92i2wsmq82u8216f7&st=9as262cc&dl=0) and extract it to `/data` folder in your container.

Run the same steps as above to generate final mesh for evaluation. 


# Baselines

We also provide the container and scripts to run HPNet, SEDNet and ComplexGen. Please refer to README.md in the `baselines` folder

<!-- ## Training

## Data preparation -->

