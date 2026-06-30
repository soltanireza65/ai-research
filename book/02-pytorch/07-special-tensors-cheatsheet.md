# Special Tensors — Cheatsheet

Quick review of [Special Tensors](07-special-tensors.md).

---

## Core ideas

- **Factories** create tensors without input data (`eye`, `arange`, `linspace`, `randn`)
- `eye(n)` — identity; `arange` — integer steps; `linspace` — evenly spaced floats
- `randn` = standard normal (weight init); `rand` = uniform [0, 1)
- `torch.manual_seed(k)` fixes CPU random sequence for reproducibility
- Masks: `torch.tril(torch.ones(n, n))` for causal patterns

---

## Python

```python
import torch

I = torch.eye(3)
pos = torch.arange(128)              # token positions
grid = torch.linspace(0, 1, 5)       # includes endpoints

torch.manual_seed(42)
W = torch.randn(512, 512) * 0.02
mask = torch.tril(torch.ones(8, 8, dtype=torch.bool))
```

---

## Preview / payoff terms

| Term | One line | Learn in |
|------|----------|----------|
| Sinusoidal PE | Position angles from `arange` + sin/cos | [Decoder-Only Transformer](../04-transformers/04-decoder-only-transformer.md) |
| Embedding lookup | Index rows of embedding matrix | [Self-Attention](../04-transformers/02-self-attention.md) |

---

## Stuck?

Reread §3 (factory definitions) and the factory table in §4 of the [full chapter](07-special-tensors.md).

---

→ [Full chapter](07-special-tensors.md) · [Vocabulary Roadmap](../00-intro/04-vocabulary-roadmap.md)
