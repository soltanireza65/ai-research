# Creating Tensors — Cheatsheet

Quick review of [Creating Tensors](01-creating-tensors.md).

---

## Core ideas

- A **tensor** is a multi-dimensional array with **shape**, **dtype**, and **device**
- **Rank** = number of dimensions; **numel** = product of shape
- Default ML dtype: `float32`; labels/indices often `int64`
- `torch.tensor()` copies; `torch.from_numpy()` shares memory
- `requires_grad=True` lets autograd track gradients (preview: backprop chapter)

---

## Python

```python
import torch

x = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float32)
z = torch.zeros(2, 3, dtype=torch.float32)
r = torch.randn(4)
print(x.shape, x.dtype, x.numel())
```

---

## Preview / payoff terms

| Term | One line | Learn in |
|------|----------|----------|
| Embeddings | Vector per token/category | [Vectors](../01-math/03-vectors.md) |
| Autograd | Tracks ops for gradients | [Backpropagation](../03-neural-networks/03-backpropagation.md) |
| Transformer | Attention-based sequence model | [Decoder-Only Transformer](../04-transformers/04-decoder-only-transformer.md) |

---

## Stuck?

Reread §3 (shape, dtype, device) and §4 (factories) in the [full chapter](01-creating-tensors.md).

---

→ [Full chapter](01-creating-tensors.md) · [Vocabulary Roadmap](../00-intro/04-vocabulary-roadmap.md)
