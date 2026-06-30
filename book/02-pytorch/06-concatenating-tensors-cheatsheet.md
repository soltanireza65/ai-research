# Concatenating Tensors — Cheatsheet

Quick review of [Concatenating Tensors](06-concatenating-tensors.md).

---

## Core ideas

- **`torch.cat`** joins along an **existing** dimension — that dim grows
- **`torch.stack`** creates a **new** dimension — all shapes must match exactly
- Sequence join: `cat([a, b], dim=1)` for `(B, T1, D)` + `(B, T2, D)`
- Batch from samples: `stack(samples, dim=0)`
- Wrong `dim` merges batch with sequence — always sketch shapes first

---

## Python

```python
import torch

a = torch.randn(2, 3)
b = torch.randn(2, 3)
torch.cat([a, b], dim=0)     # (4, 3)
torch.stack([a, b], dim=0)   # (2, 2, 3)

prefix = torch.randn(2, 5, 128)
suffix = torch.randn(2, 10, 128)
seq = torch.cat([prefix, suffix], dim=1)  # (2, 15, 128)
```

---

## Preview / payoff terms

| Term | One line | Learn in |
|------|----------|----------|
| Multi-head merge | Concat head outputs before `W_O` | [Multi-Head Attention](../04-transformers/03-multi-head-attention.md) |
| U-Net skip cat | Encoder + decoder features on channels | Vision architecture docs; skill here is `cat(..., dim=1)` |

---

## Stuck?

Reread §2 (cat vs stack intuition) and the shape table in §3 of the [full chapter](06-concatenating-tensors.md).

---

→ [Full chapter](06-concatenating-tensors.md) · [Vocabulary Roadmap](../00-intro/04-vocabulary-roadmap.md)
