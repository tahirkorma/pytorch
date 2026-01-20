# PyTorch Fundamentals & Deep Learning Projects (ANN / CNN / RNN)

This repository is a hands-on collection of **PyTorch notebooks** covering the core fundamentals of PyTorch along with practical deep learning projects implemented using:

- **ANN (Artificial Neural Networks)**
- **CNN (Convolutional Neural Networks)**
- **RNN / LSTM (Recurrent Neural Networks)**
- Training optimization techniques
- GPU acceleration (CUDA)
- Hyperparameter optimization using Optuna

The goal of this repo is to build a strong foundation in PyTorch—from tensors and autograd to complete training pipelines—and then apply that knowledge to real deep learning model implementations.

---

## 📌 What this repository contains

### ✅ PyTorch Core Fundamentals
These notebooks focus on the essential building blocks:

- Working with **PyTorch tensors**
- Understanding **autograd**
- Building training loops and reusable pipelines
- Learning how to scale training using GPU

---

### ✅ Training Pipeline & Best Practices
These notebooks focus on building robust training workflows:

- Custom training pipeline
- Using `Dataset` and `DataLoader`
- Using `torch.nn.Module` properly

---

### ✅ Deep Learning Projects Included

#### 🔹 ANN Project (Fashion-MNIST)
- ANN model implementation on Fashion-MNIST
- GPU-enabled training version
- Optimized version
- Optuna-based hyperparameter optimization

#### 🔹 CNN Project (Fashion-MNIST)
- CNN implementation for Fashion-MNIST
- GPU optimization

#### 🔹 RNN / LSTM Projects (NLP)
- QA system using RNN
- Next-word prediction using LSTM

#### 🔹 Transfer Learning
- Transfer learning based CNN + ANN workflow on Fashion-MNIST with GPU support

---

## 📂 Notebooks in this repo

| Notebook | Description |
|---------|-------------|
| `tensors_in_pytorch.ipynb` | Tensor creation & operations |
| `autograd_in_pytorch.ipynb` | PyTorch autograd fundamentals |
| `training_pipeline_in_pytorch.ipynb` | Custom training pipeline from scratch |
| `training_pipeline_using_dataset_and_dataloader_in_pytorch.ipynb` | Training pipeline using `Dataset` & `DataLoader` |
| `training_pipeline_using_nn_module_in_pytorch.ipynb` | Training pipeline using `nn.Module` |
| `ann_fashion_mnist_in_pytorch.ipynb` | ANN on Fashion-MNIST |
| `ann_fashion_mnist_gpu_in_pytorch.ipynb` | ANN on GPU |
| `ann_fashion_mnist_gpu_optimized_in_pytorch.ipynb` | Optimized GPU ANN |
| `ann_fashion_mnist_gpu_optimized_using_optuna_in_pytorch.ipynb` | Optuna tuning + ANN |
| `cnn_fashion_mnist_gpu_optimized_in_pytorch.ipynb` | CNN model (GPU optimized) |
| `transfer_learning_cnn__ann_fashion_mnist_gpu_in_pytorch.ipynb` | Transfer learning workflow |
| `qa_using_rnn_in_pytorch.ipynb` | QA task using RNN |
| `next_word_predictor_using_lstm_in_pytorch.ipynb` | Next word prediction using LSTM |

---

## ⚙️ Requirements

This repository is built around Jupyter notebooks and the PyTorch ecosystem.

### Recommended environment
- Python 3.8+
- PyTorch
- torchvision
- numpy, pandas
- matplotlib
- scikit-learn
- tqdm
- optuna (for hyperparameter tuning notebook)
