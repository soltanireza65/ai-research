# Diffusion Models

> **Status:** Stub chapter — Part II placeholder. Complete Part I (math through transformers) before expanding this chapter.

## Overview

Diffusion models power modern image, audio, and video generators — including systems like Stable Diffusion and DALL·E 3. Unlike autoregressive transformers that predict one token at a time, diffusion models learn to **reverse a noise process**: start from random static and iteratively denoise until a coherent sample emerges.

This chapter will build on probability (Module 01), gradient-based optimization (derivatives and gradients), and neural network function approximators (Module 03). You will learn the forward noising process, the reverse denoising network, training objectives (often simplified to predicting noise), and sampling schedules.

## Planned Topics

- Forward process: gradually adding Gaussian noise to data
- Reverse process: learning to undo noise step by step
- Score matching and denoising score matching intuition
- Connection to variational objectives (ELBO sketch)
- Classifier-free guidance and conditioning (text-to-image)
- Minimal PyTorch demo: train a tiny diffusion model on 2D points

## Prerequisites

Complete through `book/04-transformers/04-decoder-only-transformer.md` and `book/01-math/06-probability.md`. Comfort with PyTorch training loops from Module 03 is assumed.

## Further Reading (when ready)

- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [What are Diffusion Models? — Lil'Log](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

Return to [Learning Path](../00-intro/03-learning-path.md) for how this fits the full curriculum.
