# Transposing Tensors — Cheatsheet

Quick review of [Transposing Tensors](03-transposing-tensors.md).

---

## Core ideas

- **Transpose** swaps rows and columns (2D) or swaps dimension pairs
- `.T` on 2D only; on 3D+ use `.transpose(d0, d1)` or `.permute(...)`
- `(AB)^T = B^T A^T`
- Transpose can make tensors **non-contiguous** — `.contiguous()` before `.view()`
- `nn.Linear` needs `x @ W.T` because weights are `(out, in)`

---

## Python

```python
import torch

A = torch.randn(2, 3)
A.T                          # (3, 2)

K = torch.randn(2, 8, 64)
Kt = K.transpose(-2, -1)     # (2, 64, 8) — swap last two axes
scores = torch.randn(2, 8, 64) @ Kt
```

---

## Preview / payoff terms

| Term | One line | Learn in |
|------|----------|----------|
| Attention `K.T` | Align Q rows with K columns for scores | [Self-Attention](../04-transformers/02-self-attention.md) |
| Backprop through transpose | Gradient transposes again in backward | [Backpropagation](../03-neural-networks/03-backpropagation.md) |

---

## Stuck?

Reread §3 (`.T` vs `.transpose`) and print shapes before/after each op in the [full chapter](03-transposing-tensors.md).

---

→ [Full chapter](03-transposing-tensors.md) · [Vocabulary Roadmap](../00-intro/04-vocabulary-roadmap.md)
