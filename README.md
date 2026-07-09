# PixelRNN Image Completion

## Overview

This project implements a **Pixel Recurrent Neural Network (PixelRNN)** for image completion. The model reconstructs missing regions of partially occluded images by predicting pixels sequentially while conditioning each prediction on previously generated pixels.

---

# PixelRNN Image Completion Report

## 1. Introduction

The objective of this assignment was to implement and analyze a **Pixel Recurrent Neural Network (PixelRNN)** for image completion.

PixelRNN models image generation as an autoregressive process where each pixel is predicted based on the previously generated pixels. This enables the model to capture spatial dependencies and reconstruct missing regions in images with realistic textures and structures.

The project involved:

- Implementing the PixelRNN architecture
- Training the model on the CIFAR-10 dataset
- Evaluating reconstruction performance using training metrics and qualitative visual results

---

## 2. Methodology

### 2.1 Dataset

The experiment was conducted using the **CIFAR-10** dataset, which contains **60,000 RGB images** across **10 object classes**. Each image has a resolution of **32 × 32 pixels**.

To simulate image completion, random rectangular regions of each image were masked, and the model was trained to reconstruct the missing pixels.

| Dataset | Number of Images | Image Size |
|---------|-----------------:|-----------|
| Training | 50,000 | 32 × 32 × 3 |
| Validation | 10,000 | 32 × 32 × 3 |

---

### 2.2 Preprocessing

The following preprocessing steps were applied:

- Normalize pixel values to the **[0, 1]** range.
- Apply random rectangular masks to simulate missing regions.
- Convert images into PyTorch tensors.
- Apply optional data augmentation including:
  - Random horizontal flips
  - Random cropping

---

### 2.3 Model Architecture

The PixelRNN architecture consists of stacked masked convolution and recurrent layers that preserve the autoregressive property of image generation.

The major components include:

- **Masked Convolution (Type A):** Prevents access to future pixels during prediction.
- **LSTM Blocks:** Capture long-range spatial dependencies across the image.
- **Masked Convolution (Type B):** Produces final RGB pixel predictions.

| Layer | Description | Output Shape |
|-------|-------------|--------------|
| Input | RGB Image | (32, 32, 3) |
| Masked Conv (Type A) | 7×7 Filters | (32, 32, 64) |
| LSTM Block ×2 | Recurrent Pixel Dependencies | (32, 32, 64) |
| Masked Conv (Type B) | Output RGB Prediction | (32, 32, 3) |

---

## 3. Results

### 3.1 Training Performance

The model was trained for **10 epochs** using:

- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Loss Function:** Mean Squared Error (MSE)

Training and validation loss decreased steadily throughout training, demonstrating effective learning. Structural Similarity Index (SSIM) also improved over epochs, indicating better reconstruction quality.

### Training & Validation Curves

> **Loss Graphs per epoch **


<img width="887" height="391" alt="Screenshot 2026-07-10 at 12 37 33 AM" src="https://github.com/user-attachments/assets/19d172b4-9779-40a5-9404-6861a06c63db" />


---

### 3.2 Visual Reconstructions

The following examples compare the occluded input, model reconstruction, and ground truth image.


<img width="750" height="375" alt="image" src="https://github.com/user-attachments/assets/e1e19dda-5ef2-459b-8449-e4c265638f4b" />


The model successfully reconstructed missing regions while preserving object boundaries and maintaining color consistency.

---

## 4. Discussion

Training demonstrated that PixelRNN progressively improved its reconstruction capabilities.

During the initial epochs, the generated outputs appeared blurry due to insufficient contextual understanding. As training progressed, the model learned to:

- Preserve local texture consistency.
- Infer global object structure from surrounding pixels.
- Generate smoother pixel transitions.
- Produce visually coherent image completions.

### Challenges Encountered

- High computational cost due to sequential pixel prediction.
- Long training time compared to convolutional architectures.
- Sensitivity to masking strategy and mask size.
- Limited parallelization because of autoregressive dependencies.

### Key Insights

- SSIM provided a better indication of visual quality than MSE alone.
- Smaller receptive fields preserved finer image details.
- Model depth and masking strategy significantly influenced reconstruction performance.
- Longer training improved texture consistency and structural accuracy.

---

## 5. Conclusion

This project successfully implemented a **PixelRNN** model for image completion using the CIFAR-10 dataset.

The model learned meaningful pixel dependencies and demonstrated the ability to reconstruct missing image regions with visually coherent outputs. Throughout training, reconstruction loss decreased steadily while SSIM improved, indicating effective learning.

Although PixelRNN produces high-quality reconstructions, its sequential nature results in higher computational cost and slower inference compared to modern architectures.

### Future Improvements

- Implement **PixelCNN++** for faster parallel inference.
- Integrate attention mechanisms for improved contextual understanding.
- Train on larger datasets such as **CelebA** or **ImageNet**.
- Experiment with deeper recurrent architectures and alternative masking strategies.
- Evaluate additional perceptual metrics such as PSNR and LPIPS.

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Torchvision
- Matplotlib
- OpenCV

---

## Repository Structure

```text
.
├── checkpoints/
├── conv/
├── edges2shoes/
├── outputs/
├── dataset.py
├── model.py
├── train.py
├── server.py
├── README.md
└── requirements.txt
```

---

## Author

**Sajad Ahmed**
