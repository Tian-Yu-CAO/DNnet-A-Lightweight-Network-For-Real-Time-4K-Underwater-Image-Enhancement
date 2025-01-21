# DNnet: dynamic range and average normalization network

This repository is the official PyTorch implementation of DNnet: A lightweight network for real-time 4K underwater image enhancement using dynamic range and average normalization. [You can view the pre-proof version of our paper here.](https://www.sciencedirect.com/science/article/pii/S0957417425001836)  

## 🚀 About DNnet
DNnet utilizes a dynamic range and average normalization technique to significantly improve the quality of underwater images captured in 4K resolution. The model is optimized for real-time processing, making it suitable for practical applications in marine research, underwater robotics, and other related fields. The key features of our approach are as follows:

* **Dynamic Range and Average Normalization:**
 
<img src="https://raw.githubusercontent.com/Tian-Yu-CAO/DNnet-A-Lightweight-Network-For-Real-Time-4K-Underwater-Image-Enhancement/main/Features/Structure.jpg" width="550" height="500">

* **Lightweight Architecture:** 

<img src="https://raw.githubusercontent.com/Tian-Yu-CAO/DNnet-A-Lightweight-Network-For-Real-Time-4K-Underwater-Image-Enhancement/main/Features/Efficiency.jpg" width="600" height="200">

* **Real-Time 4K Image Enhancement:**

<img src="https://raw.githubusercontent.com/Tian-Yu-CAO/DNnet-A-Lightweight-Network-For-Real-Time-4K-Underwater-Image-Enhancement/main/Features/Performance.jpg" width="600" height="300">

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



