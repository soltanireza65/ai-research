# Multi-Head Attention

## 1. Introduction

A single self-attention head computes one \(n \times n\) attention pattern — one way to relate tokens. **Multi-head attention** runs \(h\) independent attention operations **in parallel**, each with smaller dimension \(d_{head} = d_{model} / h\), then **concatenates** the results and applies a final linear projection.

The intuition: one head might track syntax (subject-verb agreement), another coreference ("it" → noun), another local bigrams. Multiple heads let the model capture heterogeneous relationships without one attention matrix doing everything.

After this chapter you will be able to:

- Reshape projected Q, K, V into \((\text{batch}, h, n, d_{head})\).
- Run scaled dot-product attention per head independently.
- Concatenate head outputs and project with \(\mathbf{W}_O\).
- Relate the implementation to `nn.MultiheadAttention` and transformer papers.

**Where this appears in AI:** The original transformer uses \(h=8\) heads with \(d_{model}=512\). GPT-3 uses 96 heads with \(d_{model}=12288\). Every production LLM uses multi-head (or multi-query / grouped-query variants that share keys and values across heads for efficiency).

---

## 2. Intuition

> 💡 Intuition
>
> Imagine eight detectives examining the same crime scene. Each detective looks for different clues — fingerprints, alibis, timelines, witnesses. They do not share notes during inspection (separate heads). Afterward, a lead detective merges all reports into one summary (concatenation + output projection). Multi-head attention is parallel specialized inspection, then fusion.

```
  X ──► [Head 1: Q₁K₁V₁] ──► out₁ ──┐
    ──► [Head 2: Q₂K₂V₂] ──► out₂ ──┼── concat ──► W_O ──► final output
    ──► [Head h: ...    ] ──► out_h ──┘
```

Each head has its own \(\mathbf{W}_Q^{(i)}, \mathbf{W}_K^{(i)}, \mathbf{W}_V^{(i)}\) (often implemented as one large matrix, then split). Heads do not interact until concatenation.

The point is not that every head will learn a clean human-named skill. Real transformer heads are messier than textbook diagrams. But the architectural bias is still useful: each head receives a different learned projection of the same tokens, so each head gets a different similarity space. One head can make two tokens look similar while another head makes those same tokens look unrelated. The model does not have to choose a single definition of relevance.

This matters for language because relevance is multi-dimensional. In the sentence "The programmer fixed the bug because it broke tests," the word "it" may need one relation for coreference, another for syntax, and another for local phrase structure. Multi-head attention gives the neural network several parallel routes for these relationships before the output projection recombines them.

> 🔬 Deep Dive
>
> Head dimension \(d_{head} = d_{model} / h\) keeps total compute roughly similar to single-head attention with full \(d_{model}\): \(h\) matrices of size \(d_{model} \times d_{head}\) vs one \(d_{model} \times d_{model}\) projection. Parameter count is comparable; representational diversity increases.

The output projection \(\mathbf{W}_O\) is not just a cleanup step. Concatenation places head outputs side by side, but side by side vectors do not yet interact. \(\mathbf{W}_O\) lets the model mix information across heads and rotate the combined representation back into the model's shared residual stream. Without this projection, the next layer would receive separated head channels with less opportunity to combine them immediately.

---

## 3. Formal Definitions

### Hyperparameters

- \(d_{model}\) — model dimension
- \(h\) — number of heads
- \(d_{head} = d_{model} / h\) — dimension per head (must divide evenly)

### Combined projections (efficient form)

\[
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{W}_Q \in \mathbb{R}^{d_{model} \times d_{model}}
\]

Same for \(\mathbf{K}, \mathbf{V}\). Then **reshape and transpose**:

\[
\mathbf{Q} \rightarrow \mathbb{R}^{h \times n \times d_{head}}
\]

(per batch dimension omitted for clarity)

### Per-head attention

For head \(i\):

\[
\text{head}_i = \text{Attention}(\mathbf{Q}^{(i)}, \mathbf{K}^{(i)}, \mathbf{V}^{(i)})
= \text{softmax}\left(\frac{\mathbf{Q}^{(i)} {\mathbf{K}^{(i)}}^\top}{\sqrt{d_{head}}}\right) \mathbf{V}^{(i)}
\]

Shape per head: \((n \times d_{head})\).

### Concatenation and output projection

\[
\text{MultiHead}(\mathbf{X}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \, \mathbf{W}_O
\]

\[
\mathbf{W}_O \in \mathbb{R}^{d_{model} \times d_{model}}
\]

Final output shape: \((n \times d_{model})\).

Shape preservation is important. Multi-head attention may split the representation into heads internally, but it returns to \(d_{model}\) before leaving the module. That means a transformer block can be stacked repeatedly: the next block receives the same kind of tensor shape as the previous block. This is the same software-engineering idea as a stable interface. Internally the module performs several reshapes and batched matrix multiplications; externally it behaves like a function from token representations to updated token representations.

| Tensor | Shape (no batch) |
|--------|------------------|
| \(\mathbf{X}\) | \((n, d_{model})\) |
| \(\mathbf{Q}\) after split | \((h, n, d_{head})\) |
| Per-head weights | \((h, n, n)\) |
| Per-head output | \((h, n, d_{head})\) |
| Concatenated | \((n, d_{model})\) |

---

## 4. Programming Perspective

```python
import torch
import torch.nn.functional as F

seq_len, d_model, num_heads = 4, 8, 2
d_head = d_model // num_heads
x = torch.randn(1, seq_len, d_model)

# Simplified: use same Q for K,V as lab starting point
Q = x.view(1, seq_len, num_heads, d_head).transpose(1, 2)
K = Q.clone()
V = Q.clone()

scores = Q @ K.transpose(-2, -1) / (d_head ** 0.5)
weights = F.softmax(scores, dim=-1)
out = weights @ V
print("per-head weights shape:", weights.shape)  # (1, 2, 4, 4)
print("per-head output shape:", out.shape)        # (1, 2, 4, 4)
```

```python
# Concatenate heads: (batch, heads, seq, d_head) -> (batch, seq, d_model)
out_merged = out.transpose(1, 2).contiguous().view(1, seq_len, d_model)
W_o = torch.randn(d_model, d_model)
final = out_merged @ W_o
print("final output shape:", final.shape)  # (1, 4, 8)
```

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
x = torch.randn(1, 4, 8)
out, weights = mha(x, x, x, need_weights=True)
print(out.shape)           # (1, 4, 8)
print(weights.shape)       # (1, 4, 4) — averaged or combined view
```

```python
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.d_head = d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, n, d = x.shape
        Q = self.Wq(x).view(B, n, self.h, self.d_head).transpose(1, 2)
        K = self.Wk(x).view(B, n, self.h, self.d_head).transpose(1, 2)
        V = self.Wv(x).view(B, n, self.h, self.d_head).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / (self.d_head ** 0.5)
        A = F.softmax(scores, dim=-1)
        heads = A @ V
        merged = heads.transpose(1, 2).contiguous().view(B, n, d)
        return self.Wo(merged), A

mha = MultiHeadAttention(8, 2)
out, A = mha(torch.randn(2, 4, 8))
print(out.shape, A.shape)
```

```python
import numpy as np

# Verify d_head scaling differs from full d_model
d_model, h = 64, 8
d_head = d_model // h
print(f"d_model={d_model}, heads={h}, d_head={d_head}")
print("Scale factor:", np.sqrt(d_head))
```

---

## 5. Visualizations

Plot per-head attention matrices side by side to see specialization (with trained models) or independent random patterns (at init).

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

seq_len, d_model, num_heads = 6, 8, 2
d_head = d_model // num_heads
torch.manual_seed(1)
x = torch.randn(1, seq_len, d_model)
Q = x.view(1, seq_len, num_heads, d_head).transpose(1, 2)
K = Q.clone()
V = Q.clone()
scores = Q @ K.transpose(-2, -1) / (d_head ** 0.5)
weights = F.softmax(scores, dim=-1).detach().numpy()

fig, axes = plt.subplots(1, num_heads, figsize=(10, 4))
for i in range(num_heads):
    axes[i].imshow(weights[0, i], cmap="viridis", vmin=0, vmax=1)
    axes[i].set_title(f"Head {i}")
    axes[i].set_xlabel("key")
    axes[i].set_ylabel("query")
plt.suptitle("Per-head attention patterns")
plt.tight_layout()
plt.show()
```

**Reading the plots:** Each head has its own \((n \times n)\) pattern. They may look different even with \(K=Q\) in the simplified lab because head slices differ.

In a trained model, these plots can reveal useful hints, but they should be interpreted carefully. A head that attends strongly to punctuation, previous tokens, or matching brackets may be doing something meaningful, yet the final prediction also depends on MLP layers, residual connections, normalization, and later attention heads. Attention visualization is a debugging lens, not a complete explanation of the model.

```python
# Parameter count: multi-head vs single
d_model, h = 512, 8
params_qkv = 3 * d_model * d_model
params_o = d_model * d_model
print("Total MHA params:", params_qkv + params_o)
print("Per-head slice dim:", d_model // h)
```

---

## 6. Worked Examples

### Example 1: Reshape walkthrough

\(d_{model}=8\), \(h=2\), \(d_{head}=4\), \(n=4\).

After \(\mathbf{Q} = \mathbf{X}\mathbf{W}_Q\), shape \((4, 8)\).

Reshape to \((4, 2, 4)\) — sequence × heads × head_dim.

Transpose to \((2, 4, 4)\) — heads × sequence × head_dim.

Each of the 2 heads runs \((4 \times 4)\) attention on 4-D vectors.

### Example 2: Concatenation

Head 0 output: \((4, 4)\). Head 1 output: \((4, 4)\).

Concat along last dim → \((4, 8) = (n, d_{model})\).

### Example 3: Scaling per head

Head uses \(\sqrt{d_{head}} = \sqrt{4} = 2\), **not** \(\sqrt{d_{model}} = \sqrt{8}\). Each head's dot products live in \(d_{head}\)-dimensional space.

### Example 4: Lab completion — merge + project

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

seq_len, d_model, num_heads = 4, 8, 2
d_head = d_model // num_heads
x = torch.randn(1, seq_len, d_model)

Q = x.view(1, seq_len, num_heads, d_head).transpose(1, 2)
K, V = Q.clone(), Q.clone()
A = F.softmax(Q @ K.transpose(-2, -1) / (d_head ** 0.5), dim=-1)
heads_out = A @ V
merged = heads_out.transpose(1, 2).reshape(1, seq_len, d_model)
Wo = nn.Linear(d_model, d_model, bias=False)
output = Wo(merged)
print("merged:", merged.shape, "output:", output.shape)
```

### Example 5: Fused QKV projection

Production code often uses one linear layer `nn.Linear(d_model, 3 * d_model)` then splits:

```python
import torch.nn as nn

d_model = 64
qkv = nn.Linear(d_model, 3 * d_model, bias=False)
x = torch.randn(2, 10, d_model)
q, k, v = qkv(x).chunk(3, dim=-1)
print(q.shape, k.shape, v.shape)  # each (2, 10, 64)
```

This is mathematically identical to three separate linears but can be faster on GPU (one kernel launch). After chunking, the reshape/transpose into heads proceeds as before.

### Example 6: Why not average head outputs?

Averaging would force each head to agree on a single representation. Concatenation preserves **distinct subspaces** per head; \(\mathbf{W}_O\) then learns how to mix them. Averaging would collapse that diversity before the model can combine head specialties — reducing expressiveness for a fixed parameter budget.

---

## 7. AI Connection

> 🧠 AI Insight
>
> Attention head analysis papers show specialized roles emerge after training — some heads attend to previous token, some to sentence boundaries, some to rare words. Multi-head design is not just engineering convenience; it provides a **basis** of relationship types the model can combine.

**Grouped-query attention (GQA):** LLaMA 2/3 uses fewer K/V head groups than Q heads — shares keys/values to cut memory bandwidth during inference while keeping diverse queries.

**Multi-query attention (MQA):** One K/V head shared across all Q heads — even faster inference, used in PaLM, Falcon.

**Vision transformers:** Patches attend across heads for local texture vs global shape.

**Output projection \(\mathbf{W}_O\):** Mixes head outputs — allows the model to combine head specialties linearly before the residual connection.

---

## 8. Common Mistakes

> ⚠️ Common Mistake
>
> **`d_model` not divisible by `num_heads`.** `view()` fails or silently mis-shapes tensors. Always assert `d_model % num_heads == 0`.

> ⚠️ Common Mistake
>
> **Wrong transpose order.** Standard pattern: `view(B, n, h, d_head).transpose(1, 2)` → `(B, h, n, d_head)`. Transposing wrong axes mixes heads with sequence positions.

> ⚠️ Common Mistake
>
> **Scaling with \(d_{model}\) instead of \(d_{head}\).** Per-head attention must divide by \(\sqrt{d_{head}}\). Using \(\sqrt{d_{model}}\) over-softens attention when heads split dimension.

> ⚠️ Common Mistake
>
> **Forgetting `contiguous()` before `view` after transpose.** Use `.transpose().contiguous().view()` or `.reshape()` to avoid layout errors.

**Correct understanding:** Split Q,K,V into h heads → attend per head → concat → multiply by \(\mathbf{W}_O\).

---

## 9. Exercises

### Easy

1. If `d_model=64`, `num_heads=8`, what is `d_head`?
2. What are shapes of `weights` and `out` in the lab notebook?
3. Why concatenate heads instead of averaging them?

### Medium

4. Implement full multi-head attention with separate Wq, Wk, Wv and Wo.
5. Plot both head attention matrices for `seq_len=8`.
6. Count parameters in `nn.MultiheadAttention(512, 8)`.

### Hard

7. Show output of concat + Wo is equivalent to a single large attention if heads are constrained — why do we still use multiple heads?
8. Implement GQA with 8 query heads and 2 key/value groups.
9. Explain memory cost of storing all head attention maps during training.

### Challenge

10. **Head ablation:** Train or forward a small transformer; zero out one head's output before Wo and measure perplexity or task metric change. Report which heads matter most.

---

## 10. Mini Project

### Multi-Head Attention Module

Build a complete `MultiHeadAttention` class matching the transformer paper:

1. Input `(batch, seq, d_model)`.
2. Project Q, K, V; split into `num_heads`.
3. Scaled dot-product attention per head.
4. Concatenate; apply `Wo`.
5. Return output and stacked attention weights `(batch, heads, seq, seq)`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.d_head = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, n, d = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        shapes = (B, n, self.h, self.d_head)
        Q, K, V = [t.view(*shapes).transpose(1, 2) for t in qkv]
        scores = Q @ K.transpose(-2, -1) / (self.d_head ** 0.5)
        A = F.softmax(scores, dim=-1)
        heads = (A @ V).transpose(1, 2).reshape(B, n, d)
        return self.out(heads), A

m = MultiHeadAttention(64, 8)
y, attn = m(torch.randn(2, 10, 64))
assert y.shape == (2, 10, 64)
assert attn.shape == (2, 8, 10, 10)
print("OK")
```

<details>
<summary>Mini project checklist</summary>

- [ ] Shapes verified at each step
- [ ] Per-head weights returned
- [ ] Wo projection included
- [ ] Compared against `nn.MultiheadAttention` on same input (approximate)

</details>

---

## 11. Interview Questions

**Q1:** Why use multiple attention heads instead of one?

**A1:** Multiple heads let the model attend to different types of relationships in parallel — syntactic, semantic, positional — each in a lower-dimensional subspace. One head must compress all relationship types into a single attention map. Heads provide representational diversity; Wo mixes their outputs.

**Q2:** What is \(d_{head}\) and how is it related to \(d_{model}\)?

**A2:** \(d_{head} = d_{model} / h\) where h is the number of heads. The model dimension is split evenly across heads. Each head performs attention in \(d_{head}\)-dimensional query/key/value space. Scaling uses \(\sqrt{d_{head}}\).

**Q3:** Explain the reshape/transpose trick for batched multi-head attention.

**A3:** After linear projection, Q has shape (B, n, d_model). Reshape to (B, n, h, d_head), then transpose to (B, h, n, d_head) so batch matrix multiply applies attention independently for each head. Output is transposed and reshaped back to (B, n, d_model) before Wo.

**Q4:** What is grouped-query attention and why was it introduced?

**A4:** GQA uses fewer key/value head groups than query heads — multiple Q heads share the same K/V. It reduces KV cache size and memory bandwidth during autoregressive inference, important for long contexts and deployment, with modest quality tradeoffs compared to full multi-head.

---

## 12. Summary

### Key formulas

| Step | Formula |
|------|---------|
| Split | \(\mathbf{Q} \rightarrow (h, n, d_{head})\) |
| Per head | \(\text{softmax}(\mathbf{Q}^{(i)}{\mathbf{K}^{(i)}}^\top / \sqrt{d_{head}})\mathbf{V}^{(i)}\) |
| Merge | \(\text{Concat}(\text{heads}) \mathbf{W}_O\) |

### Key terminology

- **Multi-head attention** — parallel attention with h heads
- **d_head** — per-head dimension \(d_{model}/h\)
- **Output projection Wo** — linear mix after concatenation
- **GQA / MQA** — efficient variants sharing K/V across heads
- **contiguous().view()** — safe reshape after transpose

---

## 13. Preview

Multi-head self-attention lets every token see **all** tokens. **Decoder-only** models (GPT) must prevent tokens from seeing the **future** — otherwise next-token training would cheat. The next chapter adds the **causal mask**: an upper-triangular \(-\infty\) mask so position \(i\) attends only to positions \(\leq i\).

Full attention is for encoding. Causal attention is for generation.

**Next chapter:** [Decoder-Only Transformer](04-decoder-only-transformer.md)

---

## Lab

Companion notebook: [`app/transformers/03_multi_head_attention.ipynb`](../../app/transformers/03_multi_head_attention.ipynb)
