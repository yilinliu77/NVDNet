# NVDNet

# Install

## Docker (Recommended)

```bash
cd docker && docker build -t nvdnet .

docker run -it -d -p 6100:22 --name nvdnet -v /path/to/your/data:/data nvdnet

ssh -p 6100 root@127.0.0.1 (password: root)
```

## Manual

Please refer to `install.sh` for manual installation. This script has been tested on Ubuntu 22.04 with cuda 11.8.0.

# Usage

## Inference (Point cloud)

## Prepare data

Organize the input folder as follows:

```
- root
| - poisson
| | - xxx.ply (points)
```

Replace the `{data root}` with your data path::

```
export DATA_ROOT={data root}
export ROOT_PATH=$(pwd)
```

### Generate UDF field using NDC

This step will generate the UDF field for the input point cloud using NDC. The output will be saved in `${DATA_ROOT}/feat/ndc`. You can speed up the process if you have multiple GPUs by replacing `1` with the number of GPUs you have. You can also use "CUDA_VISIBLE_DEVICES=x" to restrict the GPU usage.

`cd ${ROOT_PATH}/NDC && python generate_test_voronoi_data.py ${DATA_ROOT} 1`

`${ROOT_PATH}/build/prepare_evaluate_gt_feature/prepare_evaluate_gt_feature ${DATA_ROOT}/ndc_mesh ${DATA_ROOT}/feat/ndc --resolution 256`


### Voronoi prediction

This step will use NVDNet to predict the Voronoi diagram for the UDF field. The output Voronoi and the visualization will be saved in `${DATA_ROOT}/voronoi`. 

`cd ${ROOT_PATH}/python && python test_model.py dataset.root={DATA_ROOT} dataset.output_dir={DATA_ROOT}/voronoi`

### Primitive extraction

This step will extract the mesh from the Voronoi diagram. The output mesh will be saved in `${DATA_ROOT}/mesh`.

`${ROOT_PATH}/build/extract_mesh/extract_mesh `

### Evaluation



### Baselines

Please refer to README.md in the `baselines` folder

## Training

## Data preparation

# Citation
