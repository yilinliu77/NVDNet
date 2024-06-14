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

Save the root path:

```
export ROOT_PATH={root path}
export ROOT_PATH={root path}
```

### Generate UDF field using NDC

`cd NDC && python generate_test_voronoi_data.py ${root path} {num gpus} && cd ..`

`./build/prepare_evaluate_gt_feature/prepare_evaluate_gt_feature {root path}/ndc_mesh {root path}/feat/ndc --resolution 256`

Replace the `{root path}` and `{num gpus}` with your data. Note that `prepare_evaluate_gt_feature` will automatically detect all the local GPU to calculate the UDF. You can use "CUDA_VISIBLE_DEVICES=x" to restrict the GPU usage.

### Voronoi prediction

`python test_model.py dataset.root={root path} dataset.output_dir={root path}/voronoi`

### Primitive extraction

`./build/extract_mesh/extract_mesh `

### Evaluation



### Baselines

Please refer to README.md in the `baselines` folder

## Training

## Data preparation

# Citation
