# Deep MMD Gradient Flow & MMD-Guided Diffusion Models

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

**Train and sample from generative models using Maximum Mean Discrepancy (MMD) without adversarial training**

---

## 📌 Overview

This repository implements two cutting-edge approaches in generative modeling:

1. **Deep MMD Gradient Flow (DMMD)**  
   A training algorithm using MMD-based gradient flows for stable, interpretable generation 

2. **MMD-DDM: Fast Diffusion Inference**  
   Accelerates sampling in diffusion models via MMD fine-tuning 

### Key Features
✅ Unified PyTorch framework  
✅ Non-adversarial training  
🚀 Faster sampling than standard diffusion  
📊 Built-in evaluation metrics (KID, MMD, 3-sample tests)  

---

## 🏗️ Features

| Category       | Features |
|----------------|----------|
| **Training**   | DMMD training, Learnable kernels, Custom datasets |
| **Inference**  | MMD-guided sampling, Adaptive step scheduling |
| **Evaluation** | KID, MMD, 3-sample tests, Visualization tools |
| **Supported**  | CIFAR-10, CelebA, Toy datasets |

---

📜 Citation
```bash
@article{xu2024deep,
  title={Deep MMD Gradient Flow},
  author={Xu, Bowen and Gretton, Arthur and Sutherland, Dan},
  journal={arXiv preprint arXiv:2402.17407},
  year={2024}
}

@article{wang2023mmd,
  title={MMD-DDM: Fast Inference in Denoising Diffusion Models via MMD Finetuning},
  author={Wang, Fanbo and Yang, Zhongkai and others},
  journal={arXiv preprint arXiv:2306.17189},
  year={2023}
}
```

## 🚦 Current Status

- **Latest Development Focus**:
  - Optimizing memory usage for large-scale datasets
  - Adding MNIST support
  - Improving documentation and examples

- **Planned Features**:
  - Multi-GPU support
  - Additional kernel options
  - More comprehensive benchmarking
 
  🚧 *This project is currently under active development. API may change and features are being added regularly.*

