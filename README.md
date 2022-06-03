# Pixel2Mesh
Reimplemetation of Pixel2Mesh (ECCV 2018) by Team 19 of CS492(A), KAIST 2022 Spring Semester

## Prerequisites

### NVIDIA Driver

Check if already installed:
```sh
nvidia-smi
```

If not installed, follow the [installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#driver-installation)

### Conda

Check if already installed:
```sh
conda --version
```

If not installed, follow [miniconda installation guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

## Setting Up the Environment

```sh
conda env create -f environment.yml
conda activate p2m
```

## Our Experiment Setting

These aren't necessaraily required, but for your information.

* Graphics: GEFORCE RTX 3090
* OS: Ubuntu 20.04.2 LTS (GNU/Linux 5.4.0-77-generic x86_64)
