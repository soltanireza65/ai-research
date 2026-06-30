# Decoder-Only Transformer — Cheatsheet

Quick review of [Decoder-Only Transformer](04-decoder-only-transformer.md).

> **Payoff chapter:** **Cross-entropy** from [Probability](../01-math/06-probability.md) trains next-token prediction. **AdamW** + `loss.backward()` from [Backpropagation](../03-neural-networks/03-backpropagation.md) update weights. **Masked multi-head attention** from [Multi-Head Attention](03-multi-head-attention.md) is the core block.

---

## Core ideas

- **GPT-style** architecture: token embeddings + position embeddings → stacked blocks
- Each block: masked multi-head self-attention → MLP, with residuals + layer norm
- **Causal mask** — position \(i\) may only attend to positions \(\leq i\) (no future peeking)
- **Autoregressive** — predict next token, feed it back as input
- Training loss: **cross-entropy** on next-token prediction (shifted labels)

---

## Shape table

| Tensor | Shape | Meaning |
|--------|-------|---------|
| Token IDs | \((B, n)\) | Integer indices into vocabulary |
| Embeddings | \((B, n, d_{model})\) | Token + position vectors |
| Attention weights | \((B, h, n, n)\) | Causal: upper triangle is zero |
| Logits | \((B, n, V)\) | Raw scores over vocabulary size \(V\) |
| Targets | \((B, n-1)\) | Next tokens (input shifted by 1) |

---

## Key formulas

| Concept | Formula |
|---------|---------|
| Input embedding | \(\mathbf{x}_i = \mathbf{E}_{token}[t_i] + \mathbf{E}_{pos}[i]\) |
| Scaled score | \(s_{ij} = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}}\) |
| Causal rule | \(j > i \Rightarrow s_{ij} = -\infty\) |
| Softmax | \(p_{i,k} = e^{z_{i,k}} / \sum_\ell e^{z_{i,\ell}}\) |
| Next-token loss | \(L_i = -\log p_{i,t_{i+1}}\) |

```python
# causal mask: upper triangle = -inf
mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))

# training: shift labels by one
loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, V), targets[:, 1:].reshape(-1))
```

---

## Architecture stack

1. Token + position embeddings
2. \(N\) × (masked multi-head attention → add & norm → MLP → add & norm)
3. Output projection to vocabulary logits → `cross_entropy` loss
4. `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()` (e.g. AdamW)

---

## Payoff terms

| Term | One line | Learned in |
|------|----------|------------|
| Cross-entropy | \(-\log p_{\text{true token}}\) | [Probability](../01-math/06-probability.md) |
| `loss.backward()` | Chain rule through the graph | [Backpropagation](../03-neural-networks/03-backpropagation.md) |
| AdamW | Adam + decoupled weight decay | [Backpropagation](../03-neural-networks/03-backpropagation.md) §7 |

---

## Common mistakes

| Wrong | Right |
|-------|-------|
| Full attention during training | Must mask future tokens for next-token prediction |
| Softmax on logits in loss | `cross_entropy` expects raw logits, applies log-softmax internally |
| Confuse seq length with vocab size | Attention is \(n \times n\); logits are \(n \times V\) |

## Stuck?

Reread §4 (causal mask code) and Example 3 (label shift) in the [full chapter](04-decoder-only-transformer.md).

---

→ [Full chapter](04-decoder-only-transformer.md) · [Vocabulary Roadmap](../00-intro/04-vocabulary-roadmap.md)
