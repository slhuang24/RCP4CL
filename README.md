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

## Citation

If you find this project useful, please consider citing:

```bibtex
@article{huang2025rethinking,
  title={Rethinking copy-paste for consistency learning in medical image segmentation},
  author={Huang, Senlong and Ge, Yongxin and Liu, Dongfang and Hong, Mingjian and Zhao, Junhan and Loui, Alexander C},
  journal={IEEE Transactions on Image Processing},
  year={2025},
  publisher={IEEE}
}
```

## Acknowledgements
Our code is adapted from [UniMatch](https://github.com/LiheYoung/UniMatch/tree/main). We thank these authors for their valuable work and hope that our model can also contribute to related research.
