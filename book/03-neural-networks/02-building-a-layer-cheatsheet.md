# Building a Layer — Cheatsheet

Quick review of [Building a Layer](02-building-a-layer.md).

> **Payoff chapter:** [Matrix Multiplication](../02-pytorch/02-matrix-multiplication.md) and [Matrices](../01-math/05-matrices.md) taught `@` and shape rules. This chapter applies them to `nn.Linear`.

---

## Core ideas

- A **fully connected layer** = many neurons in parallel via matrix multiply
- One `nn.Linear` replaces thousands of per-neuron dot products
- **Batch dimension** \(B\) processes many examples at once
- Forward: linear transform → activation (e.g. ReLU)
- Parameter count: \(n_{in} \cdot n_{out} + n_{out}\)

---

## Shape table

| Tensor | Shape | Meaning |
|--------|-------|---------|
| \(\mathbf{X}\) | \((B, n_{in})\) | Batch of input vectors |
| \(\mathbf{W}\) | \((n_{out}, n_{in})\) | One row per neuron |
| \(\mathbf{b}\) | \((n_{out},)\) | One bias per neuron |
| \(\mathbf{Z}\) | \((B, n_{out})\) | Pre-activations |
| \(\mathbf{Y}\) | \((B, n_{out})\) | After activation (e.g. ReLU) |

---

## Key formulas

| Concept | Formula |
|---------|---------|
| Layer forward | \(\mathbf{Z} = \mathbf{X}\mathbf{W}^\top + \mathbf{b}\) |
| With ReLU | \(\mathbf{Y} = \max(0, \mathbf{Z})\) element-wise |
| Parameters | \(n_{in} \cdot n_{out} + n_{out}\) |

```python
import torch.nn as nn

layer = nn.Linear(in_features=4, out_features=8)
X = torch.randn(32, 4)          # batch of 32
Z = layer(X)                    # (32, 8)
Y = torch.relu(Z)
```

---

## Common mistakes

| Wrong | Right |
|-------|-------|
| Wrong weight shape | `nn.Linear` stores `W` as `(out_features, in_features)` |
| Forget batch dim | Input is `(B, n_in)`, not `(n_in,)` for batched training |
| Linear-only deep net | Stack layers need nonlinear activations between them |

## Stuck?

Reread §3 (shape table) and §4 (`x @ W.T + b`) in the [full chapter](02-building-a-layer.md).

---

→ [Full chapter](02-building-a-layer.md) · [Vocabulary Roadmap](../00-intro/04-vocabulary-roadmap.md)
