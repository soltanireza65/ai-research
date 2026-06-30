# Self-Attention — Cheatsheet

Quick review of [Self-Attention](02-self-attention.md).

> **Payoff chapter:** [Attention Mechanism](01-attention-mechanism.md) gave Q/K/V attention; this chapter derives them from the same input \(\mathbf{X}\) via learned projections.

---

## Core ideas

- **Self-attention** — Q, K, V all derived from the **same** input sequence \(\mathbf{X}\)
- Learned projections: \(\mathbf{Q} = \mathbf{X}\mathbf{W}_Q\), same for K and V
- Every token attends to every other token in **one parallel pass**
- Output mixes information → **contextual embeddings**
- Without position info, self-attention is **permutation equivariant** — add positional encodings

---

## Shape table

| Tensor | Shape (no batch) | Meaning |
|--------|------------------|---------|
| \(\mathbf{X}\) | \((n, d_{model})\) | Input embeddings |
| \(\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V\) | \((d_{model}, d_{model})\) | Learned projection weights |
| \(\mathbf{Q}, \mathbf{K}, \mathbf{V}\) | \((n, d_{model})\) | Projected queries, keys, values |
| \(\mathbf{A}\) | \((n, n)\) | Attention weights |
| Output | \((n, d_{model})\) | Context-aware representations |

With batch: \(\mathbf{X}\) is \((B, n, d_{model})\); attention weights are \((B, n, n)\).

---

## Key formulas

| Step | Formula |
|------|---------|
| Projections | \(\mathbf{Q} = \mathbf{X}\mathbf{W}_Q\), \(\mathbf{K} = \mathbf{X}\mathbf{W}_K\), \(\mathbf{V} = \mathbf{X}\mathbf{W}_V\) |
| Self-attention | \(\text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_{model}}}\right)\mathbf{V}\) |
| Residual (typical) | \(\mathbf{X} + \text{SelfAttention}(\mathbf{X})\) |

```python
class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):  # (B, n, d) or (n, d)
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        d_k = Q.shape[-1]
        w = F.softmax(Q @ K.transpose(-2, -1) / (d_k ** 0.5), dim=-1)
        return w @ V, w
```

---

## Payoff terms

| Term | One line | Learned in |
|------|----------|------------|
| Scaled dot-product attention | Q/K/V → softmax weights → blend V | [Attention Mechanism](01-attention-mechanism.md) |
| `nn.Linear` projections | Three weight matrices for Q, K, V | [Building a Layer](../03-neural-networks/02-building-a-layer.md) |
| Positional encoding | Injects token order (not in bare self-attention) | [Decoder-Only Transformer](04-decoder-only-transformer.md) |

---

## Common mistakes

| Wrong | Right |
|-------|-------|
| Self-attention without positions | Add positional encodings or RoPE for order |
| Different seq lengths in Q/K | Q and K must share sequence length for `Q @ K.T` |
| Batched code without `transpose(-2,-1)` | Use last-two dims for matmul on `(B, n, d)` |

## Stuck?

Reread §4 (`SelfAttention` module) and the shape trace in §6 Example 1 of the [full chapter](02-self-attention.md).

---

→ [Full chapter](02-self-attention.md) · [Vocabulary Roadmap](../00-intro/04-vocabulary-roadmap.md)
