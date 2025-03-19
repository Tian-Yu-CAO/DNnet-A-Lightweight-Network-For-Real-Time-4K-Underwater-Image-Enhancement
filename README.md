# DNnet: dynamic range and average normalization network

[![ESWA](https://img.shields.io/badge/ESWA-Paper-<COLOR>.svg)](https://www.sciencedirect.com/science/article/pii/S0957417425001836?via%3Dihub)
![license](https://img.shields.io/github/license/Tian-Yu-CAO/DNnet-A-Lightweight-Network-For-Real-Time-4K-Underwater-Image-Enhancement)
![stars](https://img.shields.io/github/stars/Tian-Yu-CAO/DNnet-A-Lightweight-Network-For-Real-Time-4K-Underwater-Image-Enhancement)

This repository is the official PyTorch implementation of DNnet: A lightweight network for real-time 4K underwater image enhancement using dynamic range and average normalization. [You can view the paper here.](https://www.sciencedirect.com/science/article/pii/S0957417425001836)  

## 🚀 About DNnet
DNnet utilizes a dynamic range and average normalization technique to significantly improve the quality of underwater images captured in 4K resolution. The model is optimized for real-time processing, making it suitable for practical applications in marine research, underwater robotics, and other related fields. The key features of our approach are as follows:

* **Architecture:**

<p align="center">
<img src="https://raw.githubusercontent.com/Tian-Yu-CAO/DNnet-A-Lightweight-Network-For-Real-Time-4K-Underwater-Image-Enhancement/main/Features/Structure.jpg" width="520" height="540">
</p>


* **Deployment:**

<p align="center">
<img src="https://raw.githubusercontent.com/Tian-Yu-CAO/DNnet-A-Lightweight-Network-For-Real-Time-4K-Underwater-Image-Enhancement/main/Features/Efficiency.jpg" width="600" height="200">
</p>

* **Performance:**

<p align="center">
<img src="https://raw.githubusercontent.com/Tian-Yu-CAO/DNnet-A-Lightweight-Network-For-Real-Time-4K-Underwater-Image-Enhancement/main/Features/Performance.jpg" width="600" height="300">
</p>

[Click here for 4K underwater image enhancement display!](https://tian-yu-cao.github.io/DNnet-A-Lightweight-Network-For-Real-Time-4K-Underwater-Image-Enhancement/)

[潭中鱼可百许头，皆若空游无所依😉](https://tian-yu-cao.github.io/DNnet-A-Lightweight-Network-For-Real-Time-4K-Underwater-Image-Enhancement/swim_in_air)

## Testing
This repository contains a pre-trained engine and corresponding validation files. You can use the `engine_test.py` script to evaluate our model's performance and view the test results. Some dependencies may require manual installation.

```bash
# Clone the repository
git clone https://github.com/Tian-Yu-CAO/DNnet-A-Lightweight-Network-For-Real-Time-4K-Underwater-Image-Enhancement.git

# Navigate to the project directory
cd DNnet-A-Lightweight-Network-For-Real-Time-4K-Underwater-Image-Enhancement

# Install the dependencies
pip install -r requirements.txt

# Run the test script
python engine_test.py
```
After successfully running the script, a table containing various metrics should be displayed, and a folder containing enhanced images should be generated. **Notice: Due to the use of mixed precision compilation, the performance of TensorRT engine test may suffer some loss compared to PyTorch.**


## Training
This repository now contains files to train our model. Before training, [You need to download dataset here. 提取码：uieb](https://pan.baidu.com/s/1ey1RgCdnB8xwGeu3aQ5x3g?pwd=uieb)After downloading and unzipping the dataset,if you only need to train the model you can simply replace the datasets folder. If you would also like to test engine, you need to manually replace the subfolder of datasets. You can use the `train.py` script with following command:

```bash
# Run the train script
python train.py
```
Note that we have provided a full training results file in this repository. You can find the record file in the directory ./work-dir/DNnet/UIEB/20250319/122851


## Citation
```bash
@article{CAO2025126561,
title = {DNnet: A lightweight network for real-time 4K underwater image enhancement using dynamic range and average normalization},
journal = {Expert Systems with Applications},
pages = {126561},
year = {2025},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2025.126561},
url = {https://www.sciencedirect.com/science/article/pii/S0957417425001836},
author = {Tianyu Cao and Zhibin Yu and Bing Zheng}
}
```
---

Stay tuned and feel free to check the progress in the repository. We appreciate your interest in our work!



