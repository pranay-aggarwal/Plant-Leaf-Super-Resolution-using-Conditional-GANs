# Plant Leaf Super-Resolution using Conditional GANs

## Project Overview

This project tackles the challenge of reconstructing high-resolution plant leaf images from severely degraded low-resolution inputs. The goal is to recover biologically meaningful details such as leaf veins, chlorosis patterns, and necrotic lesions while maintaining pixel-level fidelity to the original image.

A Conditional Generative Adversarial Network (cGAN) based on ESRGAN principles was developed and trained entirely from scratch to perform 4× image super-resolution, transforming 32×32 RGB images into 128×128 reconstructions.

Submitted as part of Intro to Deep Learning and Generative AI course in IITM BS degree
---

## Problem Statement

Low-resolution agricultural imagery often suffers from:

* Information loss
* Compression artifacts
* Sensor noise
* Blurred biological structures

The objective is to simultaneously denoise and upscale degraded images while preserving fine-grained plant characteristics.

Unlike traditional image enhancement tasks, the evaluation prioritizes reconstruction accuracy rather than visual realism alone. Generated textures must correspond to the true underlying structure of the input image.

---

## Model Architecture

### Generator

The generator is inspired by ESRGAN and consists of:

* Residual Dense Blocks (RDBs)
* Residual-in-Residual Dense Blocks (RRDBs)
* Global residual connections
* PixelShuffle upsampling layers
* Deep feature extraction modules

The network learns to reconstruct high-frequency image details while maintaining structural consistency.

### Conditional Discriminator

A conditional discriminator is trained alongside the generator.

Inputs:

* Upscaled low-resolution image
* Generated or real high-resolution image

This encourages the generator to produce outputs that remain faithful to the original degraded image rather than generating arbitrary textures.

---

## Training Strategy

### Data Processing

* Paired LR-HR image training
* Random horizontal flips
* Random vertical flips
* Normalization and tensor conversion

### Loss Functions

The model is trained using a weighted combination of:

#### Pixel Loss (L1)

Promotes accurate pixel reconstruction and directly aligns with the evaluation metric.

#### Perceptual Loss

Uses deep feature representations to preserve texture and structural information.

#### Adversarial Loss

Encourages realistic reconstruction of fine image details.

Combined objective:

```text
Total Loss = 100 × Pixel Loss
           + 1 × Perceptual Loss
           + 0.01 × Adversarial Loss
```

### Stabilization Techniques

* Mixed Precision Training (AMP)
* Label Smoothing
* Instance Noise
* Cosine Annealing Learning Rate Scheduler
* Early Stopping

---

## Evaluation

Model performance is evaluated using **Mean Absolute Error (MAE)** between generated images and ground-truth high-resolution images.

MAE measures pixel-wise reconstruction accuracy and heavily penalizes spatial deviations, making faithful recovery of biological structures essential.

---

## Key Features

* ESRGAN-inspired super-resolution architecture
* Conditional GAN framework
* Training from scratch without pretrained weights
* Perceptual and adversarial learning
* MAE-focused reconstruction optimization
* Mixed precision accelerated training

---

## Technologies Used

* Python
* PyTorch
* Torchvision
* OpenCV
* NumPy
* Pandas
* Matplotlib
* PIL

---

## Future Improvements

* Structural Similarity (SSIM) loss
* LPIPS perceptual metric
* Exponential Moving Average (EMA) weights
* Transformer-based super-resolution models
* Test-Time Augmentation (TTA)
* Lightweight deployment variants

---

## Author

Pranay Aggarwal

BS in Data Science and Applications, IIT Madras
