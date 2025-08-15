# Deep MMD Gradient Flow & MMD-Guided Diffusion Models

PyTorch implementation of:
- Deep MMD Gradient Flow (DMMD) for non-adversarial generative modeling
- MMD-DDM for accelerated diffusion model sampling

## Features
- Deep MMD Gradient Flow training
- MMD-guided diffusion model fine-tuning
- Multiple kernel bandwidth support
- Evaluation metrics (KID, FID, MMD)

## Installation
```bash
git clone https://github.com/zahramh99/mmd-gen
cd mmd-gen
pip install -r requirements.txt

## Training DMMD
python runners/train_dmmd.py
Fine-tuning MMD-DDM
python runners/finetune_mmd_ddm.py

## Results
CIFAR-10 Samples
https://outputs/samples_50000.png

## Performance
Method	KID (x10^-3)	FID	Sampling Time (s)
DMMD	12.4 ± 1.2	45.6	0.02
MMD-DDM (10 steps)	8.7 ± 0.9	32.1	0.15