# Indexing and Slicing — Cheatsheet

Quick review of [Indexing and Slicing](05-indexing-and-slicing.md).

---

## Core ideas

- `x[i]`, `x[:, j]`, `x[a:b]` — NumPy-style, half-open slices
- Slices are often **views** (shared memory) — clone before mutating
- Boolean mask: `x[x > 0]`; fancy index: `x[indices]`
- Last token per sequence: `hidden[:, -1, :]` → `(B, D)`
- Name axes: batch, sequence, feature — `(B, T, D)`

---

## Python

```python
import torch

x = torch.arange(20).reshape(4, 5)
row = x[1, :]
sub = x[1:3, 2:4]

hidden = torch.randn(8, 50, 256)
last = hidden[:, -1, :]               # (8, 256)

logits = torch.randn(4, 10)
labels = torch.tensor([2, 5, 1, 9])
picked = logits[torch.arange(4), labels]
```

---

## Preview / payoff terms

| Term | One line | Learn in |
|------|----------|----------|
| Causal mask | Block attention to future tokens | [Self-Attention](../04-transformers/02-self-attention.md) |
| Cross-entropy indexing | Pick true-class logit per row | [Probability](../01-math/06-probability.md) |

---

## Stuck?

Reread §4 (3D batch/seq/feature examples) in the [full chapter](05-indexing-and-slicing.md).

---

→ [Full chapter](05-indexing-and-slicing.md) · [Vocabulary Roadmap](../00-intro/04-vocabulary-roadmap.md)
