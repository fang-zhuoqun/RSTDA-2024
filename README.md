# RSTDA-2024
This is a code demo for the paper "Regularized spatial–spectral transformer for domain adaptation in hyperspectral image classification" in _Journal of Applied Remote Sensing_.

## Requirements
You can obtain the docker image via ```docker pull fzq15980/msm:0.7```

## Datasets

The hyperspectral datasets in MAT format can be downloaded from one of the following sources:

+ Figshare

Link: https://doi.org/10.6084/m9.figshare.26761831

+ BaiduYun

Link: https://pan.baidu.com/s/1jnn8N38GF6lNB18fejlvlg?pwd=13g8 Password: 13g8

+ OneDrive

Link: https://saueducn-my.sharepoint.com/:f:/g/personal/fangzhuoqun_sau_edu_cn/Eksc3Z9UJMpKvxWwx42aPKYB9rvoSxniyBNph5FyD-vnyA?e=SK93Gc Password: x893

The downloaded files should be moved to `./datasets` folder. An example dataset folder has the following structure:

```
datasets
├── Pavia
│   ├── paviaU.mat
│   └── paviaU_gt_7.mat
│   ├── pavia.mat
│   └── pavia_gt_7.mat
├── HyRANK
│   ├── Dioni_gt_out68.mat
│   └── Dioni.mat
│   ├── Loukia_gt_out68.mat
│   ├── Loukia.mat
├── YRD
│   ├── yrd_nc12_7gt.mat
│   └── yrd_nc12.mat
│   ├── yrd_nc13_7gt.mat
│   ├── yrd_nc13.mat
```

## Usage:
Take RSTDA method on the Pavia task as an example: 
1. Open a terminal or put it into a pycharm project. 
2. Put the dataset into the correct path. 
3. Run RSTDA_Pavia.py.

## Citation
If you found our project helpful, please kindly cite our paper:
```
@article{10.1117/1.JRS.18.042610,
author = {Zhuoqun Fang and Yi Hu and Zhenhua Tan and Zhaokui Li and Zhuo Yan and Yutong He and Shaoteng Luo and Ye Cao},
title = {{Regularized spatial–spectral transformer for domain adaptation in hyperspectral image classification}},
volume = {18},
journal = {Journal of Applied Remote Sensing},
number = {4},
publisher = {SPIE},
pages = {042610},
keywords = {classification, transformer, regularization, unsupervised domain adaptation, hyperspectral image, Transformers, Feature extraction, Education and training, Convolution, Adversarial training, Data modeling, Image classification, Hyperspectral imaging, Mathematical optimization, Network architectures},
year = {2024},
doi = {10.1117/1.JRS.18.042610},
URL = {https://doi.org/10.1117/1.JRS.18.042610}
}
```
