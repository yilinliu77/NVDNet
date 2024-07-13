# Environment Preparation

## Option 1: Packed Docker Container (Recommended)

Download the packed docker container including the pre-compiled environment and test data from [here](TODO) (~30GB) and load it using the following command:

```
cat NVD_baselines.tar | docker import - nvd_baselines_release:v0
```

Run the container using the following command:
```
docker run -it --shm-size 64g --gpus all -p 33666:22 --name nvdnet_baselines nvd_baselines_release:v0 /bin/bash
```

**Important**: Apply for Gurobi license, replace `/opt/gurobi/gurobi.lic` with your license. Otherwise, the evaluation of `ComplexGen` will be failed.

If you want to connect to the container using an SSH, default user and password is root:root.

## Option 2: Manual
Please refer to [ComplexGen](https://github.com/guohaoxiang/ComplexGen), [HPNet](https://github.com/SimingYan/HPNet), [SEDNet](https://github.com/yuanqili78/SED-Net), [Point2CAD](https://github.com/YujiaLiu76/point2cad) to install dependancy packages.

# ComplexGen

Following [ComplexGen Baseline](https://github.com/jialechen7/ComplexGen/blob/main/nvd_test.md) instructions.

# Point2CAD

Following [Point2CAD Baseline](https://github.com/jialechen7/point2cad/blob/main/nvd_test.md) instructions.