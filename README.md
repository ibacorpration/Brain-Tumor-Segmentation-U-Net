# 🧠 Brain Tumor Segmentation using U-Net

<p align="center">
  <img src="https://github.com/user-attachments/assets/5923c650-74db-4202-957b-d09dee1d2507" 
       alt="Brain Tumor Segmentation Results" 
       width="900"/>
</p>

Deep Learning project for **pixel-level brain tumor segmentation** from MRI images using an improved **U-Net** architecture with TensorFlow/Keras.

## 🎯 Overview
The model takes MRI brain images and generates segmentation masks identifying tumor regions.

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- NumPy / Pandas
- Matplotlib
- Scikit-learn

## 🧩 Pipeline
1. Load MRI images and masks
2. Split data into **80% Train / 10% Validation / 10% Test**
3. Resize images to **256×256**
4. Apply data augmentation to training data
5. Train an improved U-Net model
6. Evaluate using Dice, IoU, Precision, and Recall
7. Visualize segmentation predictions

## 🏗️ Model
The improved U-Net uses:
- Encoder–Decoder architecture
- Convolution + Batch Normalization + LeakyReLU blocks
- Skip connections
- Dropout to reduce overfitting
- Transposed convolutions for upsampling

**Loss:** Dice Loss  
**Optimizer:** Adamax (`learning_rate=1e-4`)  
**Maximum Epochs:** 100  
**Batch Size:** 30  
**Early Stopping:** Patience = 10

## 📊 Evaluation Metrics
- Dice Coefficient
- IoU (Intersection over Union)
- Precision
- Recall

## 📁 Dataset
The notebook uses the **LGG MRI Segmentation Dataset** with the following structure:

```text
kaggle_3m/
├── patient_x/
│   ├── image.tif
│   └── image_mask.tif
```

## 🚀 Run
Open the notebook in **Kaggle or Google Colab**, install the required dependencies, update the dataset path if needed, and run the cells sequentially.

> The trained model is saved as `improved_unet2.keras`.
