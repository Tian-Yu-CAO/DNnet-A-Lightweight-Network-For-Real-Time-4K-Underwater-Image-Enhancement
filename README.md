# DNnet: A lightweight network for real-time 4K underwater image enhancement using dynamic range and average normalization

## 🛠️ About DNnet
DNnet utilizes a dynamic range and average normalization technique to significantly improve the quality of underwater images captured in 4K resolution. The model is optimized for real-time processing, making it suitable for practical applications in marine research, underwater robotics, and other related fields.

## 🚀 Key Features

* **Real-Time 4K Image Enhancement:** Tailored for high-resolution underwater imagery.
* **Lightweight Architecture:** Designed to be efficient without compromising performance.
* **Dynamic Range and Average Normalization:** Innovative approach for enhanced image quality in challenging underwater environments.

[Click here for 4K underwater image enhancement display!](https://tian-yu-cao.github.io/DNnet-A-Lightweight-Network-For-Real-Time-4K-Underwater-Image-Enhancement/)

[潭中鱼可百许头，皆若空游无所依😉](https://tian-yu-cao.github.io/DNnet-A-Lightweight-Network-For-Real-Time-4K-Underwater-Image-Enhancement/swim_in_air)

## 📆 Current Code Status
Thank you for visiting the repository for DNnet – a lightweight network designed for real-time 4K underwater image enhancement using dynamic range and average normalization. We have uploaded a pre-trained engine that has been trained on the **UIEB** dataset. This model, along with the associated test scripts, is available for evaluation in the **Model Validation and Testing** section.

### Current Features:
- **Pre-trained Model**: The engine has been trained on the UIEB dataset and is now available for validation.
- **Test Scripts**: The testing code `engine_test.py` allows you to evaluate the performance of the pre-trained model. Simply run the script to see the model's results on the test data.

### What’s Next:
- **Progressive Updates**: This repository will be progressively updated with new versions of the model, each trained on different datasets or with new techniques. Stay tuned for future updates.
- **Upcoming Enhancements**: Future versions will include improved performance metrics, extended support for more datasets, and new testing features.
  
We are actively working on improving the model and expanding its capabilities. Please refer to the **Model Validation and Testing** section for the latest updates and test results.

We encourage you to try out the current model and provide feedback. More updates will be coming soon!

## 💻 Model Validation and Testing
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

### Test Result
After successfully running the script, a table containing various metrics should be displayed, and a folder containing enhanced images should be generated. **Notice: Due to the use of mixed precision compilation, the performance of TensorRT engine test may suffer some loss compared to PyTorch.**

---

Stay tuned and feel free to check the progress in the repository. We appreciate your interest in our work!



