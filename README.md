# dmmd-mmd-ddm-gen
Deep MMD Gradient Flow &amp; Fast MMD-Guided Diffusion Sampling
# 🌀 dmmd-mmd-ddm-gen

**Deep MMD Gradient Flow & MMD-Guided Fast Inference for Diffusion Models**  
*Train and sample from generative models using Maximum Mean Discrepancy (MMD), without adversarial training.*

---

## 📌 Overview

This repository implements and unifies two recent approaches in generative modeling:

- **[Deep MMD Gradient Flow (DMMD)](https://arxiv.org/abs/2402.17407)**  
  A training algorithm that replaces adversarial losses or score matching with **MMD-based gradient flows**, allowing stable and interpretable generation.
  
- **[MMD-DDM: Fast Inference in Diffusion Models via MMD Finetuning](https://arxiv.org/abs/2306.17189)**  
  A method that fine-tunes **sampling paths** in pretrained diffusion models to match target distributions using MMD, achieving **faster and cleaner inference**.

> We provide a unified PyTorch-based framework that allows you to:
> - Train models using DMMD
> - Sample faster using MMD-DDM
> - Customize kernels, datasets, and architectures
> - Evaluate using KID, MMD, and 3-sample tests

---

## 🔍 Why This Matters

Traditional GANs and diffusion models require adversarial training or complex score estimation. Our approach offers:

✅ **Non-adversarial training** using MMD  
⚡ **Fast sampling** via inference-time optimization  
🧠 **Customizable kernel functions** and distance metrics  
📈 **Evaluation tools** based on well-founded statistical tests

---

## 🏗️ Features

- 🧪 Deep MMD-based training (DMMD)
- 🚀 MMD-guided fast sampling (MMD-DDM)
- 🌐 Support for CIFAR-10, CelebA, synthetic toy datasets
- 🧠 Learnable and adaptive kernels
- 📊 Built-in metrics: KID, MMD, 3-sample tests
- 🧰 PyTorch-based with modular structure

---

## 🧬 Architecture
dmmd-mmd-ddm-gen/
├── dmmd/ # DMMD training
├── mmd_ddm/ # MMD-guided inference
├── kernels/ # MMD kernel functions (RBF, adaptive, etc.)
├── models/ # Generator, UNet, etc.
├── datasets/ # CIFAR-10, CelebA loaders
├── scripts/ # CLI for training, sampling, metrics
├── experiments/ # Jupyter notebooks
└── results/ # Samples, visualizations, metrics

## 🚀 Getting Started

### 1. Install dependencies
```bash
git clone https://github.com/zahramh99/dmmd-mmd-ddm-gen.git
cd dmmd-mmd-ddm-gen
pip install -r requirements.txt

2. Train with DMMD

python scripts/train_dmmd.py --dataset cifar10 --epochs 100
3. Fast sampling with MMD-DDM
python scripts/sample_fast.py --model-path ./checkpoints/model.pth --steps 25
4. Evaluate
python scripts/evaluate.py --real ./data/cifar10 --fake ./results/generated
📊 Results & Visualizations
<p align="center"> <img src="results/sample_grid.png" width="600" alt="Samples from DMMD + MMD-DDM"> </p>
🧠 Creative Contributions
This project goes beyond the original papers with:

🔁 Integrated training and inference pipeline

🔍 Learnable kernels for domain-specific optimization

⏱️ Adaptive sampling schedules

🧪 Support for MMD 3-sample and kernel two-sample tests

📈 Extensive kernel visualization tools

🧾 Citations
If you use this codebase, please consider citing:

bibtex
Copy
Edit
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
🙋‍♂️ Contributions
Pull requests, issues, and ideas are very welcome!
If you'd like to contribute an extension (e.g., new dataset, kernel, or benchmark), feel free to fork and open a PR.

📜 License
This project is licensed under the Apache 2.0 License.

🌍 Acknowledgements
Code built with PyTorch, NumPy, and Matplotlib

Inspired by the original authors of DMMD and MMD-DDM


