# InceptionFormer

> A deep learning framework for UAV LiDAR individual tree point cloud structural completion.

---

## 📄 Paper

**InceptionFormer: A Deep Learning Framework for Individual Tree Point Cloud Completion**

If you use this work in your research, please cite our paper.

```
@article{LUO2026115348,
title = {InceptionFormer: A deep learning framework for UAV LiDAR point cloud completion to improve tree parameters estimation in dense forests},
journal = {Remote Sensing of Environment},
volume = {338},
pages = {115348},
year = {2026},
issn = {0034-4257},
doi = {https://doi.org/10.1016/j.rse.2026.115348},
url = {https://www.sciencedirect.com/science/article/pii/S0034425726001185},
author = {Binhan Luo and Jian Yang and Shuo Shi and Ruilin Gan and Zhongliang Wu and Sihao Wang and Ao Wang and Lin Du and Wei Gong},
keywords = {UAV, Point cloud completion, Tree parameter estimation, Forest inventory, Deep learning}
}
```
---

## 🌲 Introduction

  In dense forests, UAV laser scanning (ULS) point clouds of individual trees often suffer from structural incompleteness in the lower trunk region due to canopy occlusion and signal attenuation. This incompleteness affects the accuracy of forest structural parameter estimation and carbon stock assessment.To address this issue, we propose **InceptionFormer**, a deep learning network designed for individual tree point cloud structural completion. The model integrates an Inception Feature Aggregation (IFA) module to extract multi-scale geometric features and a Sparse Attention (PSA) module to capture global contextual information, enabling effective learning of structural incompleteness.
To train and evaluate the model, we further construct the TreeCompletion3D dataset by collecting multiple publicly available UAV LiDAR datasets and simulating structural absence based on height, point density, and canopy base height (CBH).

---

## 🏗 Overall Architecture
<img width="4875" height="2677" alt="Figure_5" src="https://github.com/user-attachments/assets/040593f5-ed5e-4e9e-97fb-b952e3aea486" />
The overall architecture of InceptionFormer consists of three units: feature extraction, seed generator, and point generator. (b) The details of the seed generation unit. (c) The details of the Point Fractal Generator module, which is composed of a Multilayer Perceptron (MLP) and Point Sparse Attention. The overall framework of InceptionFormer consists of two main components:

- **Inception Feature Aggregation (IFA)**  

<img width="4889" height="2626" alt="Figure_6" src="https://github.com/user-attachments/assets/d9a0c365-c3cb-4df0-868c-65ce5793cdf4" />


- **Sparse Attention (PSA)**  
  <img width="2600" height="1330" alt="Figure_7" src="https://github.com/user-attachments/assets/53039c62-b4e6-4b67-ac7f-965a8b3a9e69" />


---

## 📊 Dataset

We construct the **TreeCompletion3D dataset** for individual tree point cloud completion.
You can find at https://zenodo.org/records/18899503.


## ⚙️ Environment Setup

This code has been tested under the following configuration:

- **Operating System:** Ubuntu 20.04  
- **Python:** 3.8.12  
- **PyTorch:** 1.9.0  
- **CUDA:** 11.2  

Please install the required dependencies and compile the third-party modules before running the code.

### Install Dependencies

```
pip install -r requirements.txt
```
### Compile Pytorch 3rd-party modules
please compile Pytorch 3rd-party modules ChamferDistancePytorch and mm3d_pn2. 
```
cd $InceptionFormer/utils/ChamferDistancePytorch/chamfer3D
python setup.py install

cd $InceptionFormer/utils/mm3d_pn2
python setup.py build_ext --inplace
```
### Install PointNet2 Ops Library
```
cd utils/PointNet2_ops_lib
python setup.py install
```
### Train
```
python train_TreeCompletion3D.py -c TreeCompletion3D.yaml
```
### Test
To test InceptionFormer on TreeCompletion3D benchmark, run:
```
python test_TreeCompletion3D.py -c TreeCompletion3D.yaml
```

## 📈 Results

Experimental results show that **InceptionFormer** outperforms existing methods across multiple tree species.

| Metric | Value |
|------|------|
| Chamfer Distance (L1) | **2.88** |
| Chamfer Distance (L2) | **1.96** |

Visualization results demonstrate improved trunk reconstruction, better geometric continuity, and more uniform point density.
<img width="3000" height="4367" alt="Figure_10" src="https://github.com/user-attachments/assets/58a04c91-db06-498f-ba9c-ff0afd80ebb0" />
<img width="3000" height="3557" alt="Figure_10_SplitPage" src="https://github.com/user-attachments/assets/8b211253-cd08-4025-b45a-23aa851b6664" />

## 🙏 Acknowledgements

Part of this implementation is inspired by the excellent open-source project [PointAttN](https://github.com/ohhhyeahhh/PointAttN). We sincerely thank the authors for their valuable contributions to the community.
