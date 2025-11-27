# GRN-hackathon

This repository contains implementations of two major diffusion model architectures for generative modeling on the Fashion-MNIST dataset:

1. **DDPM** (Denoising Diffusion Probabilistic Models)
2. **Score-Based Diffusion** (based on score matching)
3. **Flow Matching** (applied to Paul15 scRNA-seq data)

Both implementations include:
- Unified EMA (Exponential Moving Average) for stable training
- UNet based backbone
- FID score evaluation
- VAE baseline for benchmarking

# Comments:
1. ddpm_diffusion.ipynb works well and generates high-quality images with fast sampling, we should use this.
2. score_based_diffusion.ipynb also works but the generated image quality is low and the EDM sampler is not working for now.
3. paul15_flow_matching.ipynb implements flow matching on the Paul15 dataset (included in `data/`).
4. They all use UNet (with self-attention) as the backbone. When modeling GRN, we may consider changing this backbone.

# Reference:
The ddpm_diffusion.ipynb implementation is based on https://github.com/dome272/Diffusion-Models-pytorch
