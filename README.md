# Rethinking Copy-Paste for Consistency Learning in Medical Image Segmentation

## Getting Stared
* Installation
```
conda create -n env_name python=3.7.16
conda activate env_name
pip install -r requirements.txt
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
```

### Dataset

- ACDC: (https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC)

## Usage
* We provide train code, test code, data_split, and model for ACDC dataset.
* Please modify your dataset path in configuration files:
```
├── [Your ACDC Path]
    └── data
```
**Note: More details, including the complete training code and additional dataset information, will be coming soon.**
