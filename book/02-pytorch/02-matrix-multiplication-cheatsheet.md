# Matrix Multiplication — Cheatsheet

Quick review of [Matrix Multiplication](02-matrix-multiplication.md).

---

## Core ideas

- `@` is matrix multiply; `*` is element-wise — do not confuse them
- Shape rule: `(m, n) @ (n, p) → (m, p)` — inner dimensions must match
- Every `nn.Linear` forward pass is `x @ W.T + b`
- Batch dimension leads: `(B, m, n) @ (B, n, p) → (B, m, p)`
- Matmul dominates GPU time in large models

---

## Python

```python
import torch

A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = A @ B                    # (3, 5)

x = torch.randn(32, 512)
W = torch.randn(256, 512)
y = x @ W.T                  # (32, 256) — nn.Linear layout
```

---

## Preview / payoff terms

| Term | One line | Learn in |
|------|----------|----------|
| Attention (`Q @ K.T`) | Token–token similarity via dot products | [Self-Attention](../04-transformers/02-self-attention.md) |
| Backprop through matmul | Gradients flow through `@` in reverse | [Backpropagation](../03-neural-networks/03-backpropagation.md) |

---

## Stuck?

Reread §3 (2×2 table + sum formula) and trace shapes on paper in the [full chapter](02-matrix-multiplication.md).

---

→ [Full chapter](02-matrix-multiplication.md) · [Vocabulary Roadmap](../00-intro/04-vocabulary-roadmap.md)
