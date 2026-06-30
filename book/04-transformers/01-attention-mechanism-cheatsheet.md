# Attention Mechanism — Cheatsheet

Quick review of [Attention Mechanism](01-attention-mechanism.md).

> **Payoff chapter:** **Dot products** and **matrix multiply** from [Vectors](../01-math/03-vectors.md) and [Matrices](../01-math/05-matrices.md), and **softmax** from [Probability](../01-math/06-probability.md), come together here.

---

## Core ideas

- **Attention** = weighted sum of **values**, where weights come from **query–key** similarity
- **Query (Q)** — what am I looking for?
- **Key (K)** — what do I advertise about myself?
- **Value (V)** — what content do I contribute if selected?
- **Softmax** turns raw scores into weights that sum to 1 per row
- **Scaling** by \(\sqrt{d_k}\) prevents dot products from growing too large

---

## Shape table

| Tensor | Shape (no batch) | Meaning |
|--------|------------------|---------|
| \(\mathbf{Q}\) | \((n, d_k)\) | One query per position |
| \(\mathbf{K}\) | \((n, d_k)\) | One key per position |
| \(\mathbf{V}\) | \((n, d_v)\) | One value per position |
| \(\mathbf{S}\) | \((n, n)\) | Scaled query–key scores |
| \(\mathbf{A}\) | \((n, n)\) | Attention weights (row softmax) |
| \(\mathbf{O}\) | \((n, d_v)\) | Weighted blend of values |

With batch: prepend `(B, …)` to every shape.

---

## Key formulas

| Step | Formula |
|------|---------|
| Scores | \(\mathbf{S} = \mathbf{Q}\mathbf{K}^\top / \sqrt{d_k}\) |
| Weights | \(\mathbf{A} = \text{softmax}(\mathbf{S})\) row-wise |
| Output | \(\mathbf{O} = \mathbf{A}\mathbf{V}\) |
| Combined | \(\text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}\) |

```python
scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
weights = torch.softmax(scores, dim=-1)
output = weights @ V
```

---

## Payoff terms

| Term | One line | Learned in |
|------|----------|------------|
| Dot product | Measures vector alignment | [Vectors](../01-math/03-vectors.md) |
| Matrix multiply | `Q @ K.T` for all pairwise scores | [Matrices](../01-math/05-matrices.md) |
| Softmax | Scores → probabilities summing to 1 | [Probability](../01-math/06-probability.md) |

---

## Common mistakes

| Wrong | Right |
|-------|-------|
| Softmax over wrong dim | Apply row-wise: `dim=-1` for seq×seq scores |
| Forget \(\sqrt{d_k}\) scaling | Without it, softmax saturates (near one-hot) |
| Confuse Q/K/V roles | Q asks, K matches, V delivers content |

## Stuck?

Reread §3 (Q/K/V definitions) and the 2×2 worked example in §6 of the [full chapter](01-attention-mechanism.md).

---

→ [Full chapter](01-attention-mechanism.md) · [Vocabulary Roadmap](../00-intro/04-vocabulary-roadmap.md)
