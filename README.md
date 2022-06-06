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
git clone https://github.com/sundoufu/Pixel2Mesh.git
cd Pixel2Mesh/
conda env create -f environment.yml
conda activate pytorch3d
pip install git+https://github.com/sundoufu/PyTorchEMD.git
```

## Pretrained Checkpoints
1. Download from https://drive.google.com/drive/folders/1fXnnjMysnHf_vP3X6t2f3xv_9StFER_4?usp=sharing
2. Save them under Pixel2Mesh/checkpoints/

## Dataset
1. Download ShapeNetP2M/ from https://drive.google.com/drive/folders/131dH36qXCabym1JjSmEpSQZg4dmZVQid
2. Save them under data/ (data/ShapeNetP2M)

## Running the evaluation
```sh
# Inside Pixel2Mesh/
chmod +x ./eval.sh
# Run eval.sh in the background and redirect output logs to some files
nohup ./eval.sh 1>eval.out 2>eval.err &

# To see the progress real-time,
tail -f eval.err
```

## Our Experiment Setting

These aren't necessaraily required, but for your information.

* Graphics: GEFORCE RTX 3090
* OS: Ubuntu 20.04.2 LTS (GNU/Linux 5.4.0-77-generic x86_64)
