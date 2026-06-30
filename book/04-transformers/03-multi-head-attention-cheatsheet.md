# Multi-Head Attention — Cheatsheet

Quick review of [Multi-Head Attention](03-multi-head-attention.md).

> **Payoff chapter:** [Self-Attention](02-self-attention.md) with one head → \(h\) parallel heads, each in \(d_{head} = d_{model}/h\) dimensions, merged by \(\mathbf{W}_O\).

---

## Core ideas

- **Multiple heads** = parallel attention with different learned projections
- Each head has its own \(W_Q, W_K, W_V\) — learns different relationship patterns
- Split: \(d_{head} = d_{model} / h\) where \(h\) = number of heads
- Run attention per head, **concatenate**, then **output projection** \(\mathbf{W}_O\)
- One head = one "question type"; many heads = syntax, semantics, position, etc.

---

## Shape table

| Tensor | Shape (with batch) | Meaning |
|--------|-------------------|---------|
| \(\mathbf{X}\) | \((B, n, d_{model})\) | Input sequence |
| \(\mathbf{Q}, \mathbf{K}, \mathbf{V}\) after split | \((B, h, n, d_{head})\) | Per-head projections |
| Per-head weights | \((B, h, n, n)\) | Attention map per head |
| Concatenated | \((B, n, d_{model})\) | Heads merged along feature dim |
| Output after \(\mathbf{W}_O\) | \((B, n, d_{model})\) | Final mixed representation |

---

## Key formulas

| Step | Formula |
|------|---------|
| Split | \(\mathbf{Q} \rightarrow (h,\, n,\, d_{head})\) |
| Per head | \(\text{softmax}(\mathbf{Q}^{(i)}{\mathbf{K}^{(i)}}^\top / \sqrt{d_{head}})\mathbf{V}^{(i)}\) |
| Merge | \(\text{Concat}(\text{heads})\,\mathbf{W}_O\) |

```python
# reshape (B, n, d_model) -> (B, h, n, d_head)
q = q.view(B, n, h, d_head).transpose(1, 2)
k = k.view(B, n, h, d_head).transpose(1, 2)
v = v.view(B, n, h, d_head).transpose(1, 2)

scores = q @ k.transpose(-2, -1) / (d_head ** 0.5)
out = (softmax(scores, dim=-1) @ v).transpose(1, 2).contiguous().view(B, n, d_model)
out = out @ W_o
```

---

## Preview terms (optional)

| Term | One line | When to learn |
|------|----------|---------------|
| GQA | Fewer K/V groups than Q heads | Inference optimization — §7 preview |
| MQA | One shared K/V head | Faster inference — §7 preview |

---

## Common mistakes

| Wrong | Right |
|-------|-------|
| `d_model` not divisible by `h` | \(d_{head} = d_{model} / h\) must be integer |
| `.view()` after `.transpose()` | Call `.contiguous().view(...)` first |
| Scale by \(\sqrt{d_{model}}\) per head | Use \(\sqrt{d_{head}}\) inside each head |

## Stuck?

Reread §3 (reshape walkthrough) and the `MultiHeadAttention` class in §4 of the [full chapter](03-multi-head-attention.md).

---

→ [Full chapter](03-multi-head-attention.md) · [Vocabulary Roadmap](../00-intro/04-vocabulary-roadmap.md)
