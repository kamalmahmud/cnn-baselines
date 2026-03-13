# CNN Baselines in PyTorch

Minimal, readable computer vision baselines built with **PyTorch** and **Torchvision**.

This repository contains two image classification projects:

- **MNIST_CNN** — a family of lightweight CNN architectures for handwritten digit recognition on **MNIST**
- **ALEX_NET** — an **AlexNet-style** architecture adapted for **CIFAR-10**

The goal of this repository is to provide clean baseline implementations that are easy to read, run, and extend for experimentation.

---

## Why this project matters

This repo demonstrates practical deep learning engineering skills that are relevant for CV / ML roles:

- Implementing CNN architectures directly in PyTorch
- Building dataset pipelines with `torchvision`
- Applying normalization and augmentation for image classification
- Writing end-to-end training and evaluation loops
- Comparing baseline architectures across datasets of different complexity

---

## Repository structure

```text
cnn-baselines/
├── ALEX_NET/
│   └── alex_net.py
└── MNIST_CNN/
    ├── data.py
    ├── model.py
    └── train.py
```

---

## Projects

### 1) MNIST CNN baselines

The `MNIST_CNN` module contains multiple CNN variants for classifying grayscale `28×28` MNIST digits.

#### Implemented models

- **`MnistCNN`**
  - 2 convolutional layers
  - max pooling
  - 2 fully connected layers

- **`MnistCNN_V1`**
  - 3 convolutional layers
  - max pooling
  - adaptive average pooling
  - lightweight classification head

- **`MnistCNN_V2`**
  - deeper convolutional stack
  - adaptive average pooling
  - `1×1` convolution as the final classifier
  - currently used in `train.py`

#### Data pipeline

- Dataset: **MNIST**
- Automatic download via `torchvision.datasets.MNIST`
- Preprocessing:
  - tensor conversion
  - normalization with MNIST mean and standard deviation

#### Training setup

- Loss: **CrossEntropyLoss**
- Optimizer: **AdamW**
- Epochs: **10**
- Batch size: **64**

---

### 2) AlexNet on CIFAR-10

The `ALEX_NET` module adapts AlexNet for the smaller `32×32` CIFAR-10 image size while preserving the general multi-stage convolutional design.

#### Model highlights

- 5 convolutional layers
- ReLU activations
- max pooling
- dropout regularization
- 3-layer classifier head

#### Data pipeline

- Dataset: **CIFAR-10**
- Automatic download via `torchvision.datasets.CIFAR10`
- Training augmentation:
  - random horizontal flip
  - random crop with padding

- Evaluation preprocessing:
  - tensor conversion
  - channel-wise normalization

#### Training setup

- Loss: **CrossEntropyLoss**
- Optimizer: **AdamW**
- Learning rate: **0.001**
- Epochs: **100**
- Train batch size: **128**
- Test batch size: **100**

---

## Reported results

The following best accuracies are reported directly in the source code comments:

| Model      | Dataset  | Reported best test accuracy |
| ---------- | -------- | --------------------------: |
| `MnistCNN` | MNIST    |                  **99.28%** |
| `AlexNet`  | CIFAR-10 |                  **87.42%** |

> Note: these results are documented in code comments. Adding saved logs, checkpoints, and experiment configs would make the benchmark claims easier to reproduce end-to-end.

---

## Getting started

### 1) Clone the repository

```bash
git clone https://github.com/kamalmahmud/cnn-baselines.git
cd cnn-baselines
```

### 2) Install dependencies

Install a PyTorch build compatible with your machine, then install Torchvision.

```bash
pip install torch torchvision
```

### 3) Run training

#### MNIST

```bash
cd MNIST_CNN
python train.py
```

#### CIFAR-10 AlexNet

```bash
cd ALEX_NET
python alex_net.py
```

Datasets are downloaded automatically into `./data` when the scripts are executed.

---

## Technical highlights

- Modular separation of **data loading**, **model definition**, and **training** for the MNIST project
- Manual implementation of CNN architectures instead of relying on high-level training frameworks
- Use of **data augmentation** and **normalization** for better generalization on CIFAR-10
- Baseline-friendly code that can be extended with schedulers, checkpointing, mixed precision, or experiment tracking

---

## Potential next improvements

To make this repository even stronger for portfolio and recruiting use, the next steps would be:

- Add a `requirements.txt` or `pyproject.toml`
- Add saved checkpoints and reproducible experiment configs
- Track metrics per epoch and plot learning curves
- Add command-line arguments for model choice and hyperparameters
- Include sample predictions or confusion matrices
- Add a short comparison table across MNIST model variants

---

## Tech stack

- Python
- PyTorch
- Torchvision

---

## Summary

This repository showcases foundational deep learning work across two classic image classification benchmarks. It emphasizes **clean PyTorch implementation**, **baseline model design**, and **hands-on training workflows** for computer vision experimentation.
