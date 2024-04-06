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

### Voronoi prediction

### Primitive extraction


## Training

## Data preparation

# Citation
