# Reshaping Tensors — Cheatsheet

Quick review of [Reshaping Tensors](04-reshaping-tensors.md).

---

## Core ideas

- Reshape changes **layout**, not data — **numel** must stay the same
- `view` is zero-copy but needs contiguous memory; `reshape` may copy
- `-1` infers exactly one dimension from numel
- `unsqueeze` / `squeeze` add or remove size-1 axes
- Wrong reshape with matching numel = silent semantic bug

---

## Python

```python
import torch

x = torch.arange(12).reshape(2, 2, 3)
flat = x.flatten()                    # (12,)
batched = torch.randn(3, 224, 224).unsqueeze(0)  # (1, 3, 224, 224)

# CNN flatten keeping batch
features = torch.randn(16, 256, 7, 7)
flat = features.reshape(16, -1)       # (16, 12544)
```

---

## Preview / payoff terms

| Term | One line | Learn in |
|------|----------|----------|
| Multi-head split | `(B,T,D)` → `(B,h,T,d_head)` | [Multi-Head Attention](../04-transformers/03-multi-head-attention.md) |
| Cross-entropy reshape | Merge batch×seq for per-token loss | [Probability](../01-math/06-probability.md) |

---

## Stuck?

Reread §3 (numel invariant) and print `.shape` after every reshape in the [full chapter](04-reshaping-tensors.md).

---

→ [Full chapter](04-reshaping-tensors.md) · [Vocabulary Roadmap](../00-intro/04-vocabulary-roadmap.md)
