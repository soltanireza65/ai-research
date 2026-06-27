# Self-Attention

## 1. Introduction

In the previous chapter, Query, Key, and Value matrices were **given**. In a transformer, they are **computed from the same input sequence** \(\mathbf{X}\) through learned weight matrices. That is **self-attention**: every token generates a query ("what am I looking for?"), a key ("what do I contain?"), and a value ("what information do I pass forward?"), then attends over all tokens in the sequence.

Self-attention is what lets GPT relate the word "bank" to "river" or "money" depending on context. It runs in parallel across all positions — no sequential recurrence — which is why transformers scale so well on GPUs.

After this chapter you will be able to:

- Compute \(\mathbf{Q}\), \(\mathbf{K}\), \(\mathbf{V}\) from input embeddings \(\mathbf{X}\).
- Run full self-attention and interpret the \((\text{seq\_len} \times \text{seq\_len})\) weight matrix.
- Explain which position each query attends to most.
- Connect self-attention to `nn.MultiheadAttention` conceptually.

**Where this appears in AI:** Every transformer block begins with self-attention (or masked self-attention in decoders). BERT, GPT, LLaMA, Claude — all rely on this operation stacked dozens of times.

---

## 2. Intuition

> 💡 Intuition
>
> Picture a meeting room with everyone seated in a circle. Each person prepares three sticky notes: a **question** (query), a **name tag** (key), and a **fact sheet** (value). Everyone holds up their question and compares it to everyone else's name tags. The best matches get the most attention. Each person leaves with a summary that blends others' fact sheets, weighted by relevance. Nobody moves seats — information moves to them.

```
  Input X (seq_len × d_model)
       │
       ├── X @ Wq ──► Q ──┐
       ├── X @ Wk ──► K ──┼──► Attention(Q,K,V) ──► Output (seq_len × d_model)
       └── X @ Wv ──► V ──┘
```

All three projections share the same \(\mathbf{X}\) but use **different learned weights** \(\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V\). That lets the model learn separate roles for "searching," "being found," and "delivering content."

Self-attention is called "self" because the sequence consults itself. There is no external memory table in this basic form. The token at position 2 can read position 0, position 1, position 2, and so on, depending on whether the attention is bidirectional or masked. The important shift from older sequence models is that every pair of positions can interact in one layer. A recurrent model must pass information step by step through hidden states; self-attention can create a direct connection from the first token to the last token inside the attention matrix.

For an experienced programmer, you can think of the attention matrix as a learned routing table created at runtime. It is not a fixed adjacency list. The route depends on the current input embeddings and the learned projection matrices. Change one word in the sentence, and the attention weights can change because the queries and keys change.

> 🔬 Deep Dive
>
> Without separate projections, \(\mathbf{Q} = \mathbf{K} = \mathbf{V} = \mathbf{X}\) would force similarity and content to be the same vector. Learned projections decouple "how similar am I to you?" from "what information should I share?" — critical for expressive representations.

This decoupling is one reason transformer representations become contextual. The vector for `"bank"` before self-attention may mostly encode token identity. After self-attention, the vector can include information from `"river"` or `"loan"` depending on the surrounding tokens. The same token ID can therefore produce different hidden states in different sentences, which is essential for language understanding and for LLM behavior.

---

## 3. Formal Definitions

### Input sequence

\[
\mathbf{X} \in \mathbb{R}^{n \times d_{model}}
\]

- \(n\) = sequence length (number of tokens)
- \(d_{model}\) = embedding dimension

Row \(i\) is the embedding of token \(i\).

### Projection matrices

\[
\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d_{model} \times d_{model}}
\]

In practice, \(d_k = d_v = d_{model}\) in simple setups; multi-head attention (next chapter) splits dimensions.

### Projected Q, K, V

\[
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad
\mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad
\mathbf{V} = \mathbf{X} \mathbf{W}_V
\]

Each has shape \((n \times d_{model})\).

### Self-attention output

\[
\text{SelfAttention}(\mathbf{X}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_{model}}}\right) \mathbf{V}
\]

Same formula as scaled dot-product attention, but \(\mathbf{Q}, \mathbf{K}, \mathbf{V}\) all derive from \(\mathbf{X}\).

### Residual connection (in full transformer block)

Production models wrap attention with:

\[
\mathbf{X}' = \mathbf{X} + \text{SelfAttention}(\mathbf{X})
\]

The skip connection preserves gradient paths and original information.

| Symbol | Shape | Meaning |
|--------|-------|---------|
| \(\mathbf{X}\) | \((n, d_{model})\) | Input embeddings |
| \(\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V\) | \((d_{model}, d_{model})\) | Learned projections |
| \(\mathbf{A}\) | \((n, n)\) | Attention weights |
| Output | \((n, d_{model})\) | Updated representations |

---

## 4. Programming Perspective

```python
import torch
import torch.nn.functional as F

seq_len, d_model = 4, 8
x = torch.randn(seq_len, d_model)
Wq = Wk = Wv = torch.randn(d_model, d_model)

Q, K, V = x @ Wq, x @ Wk, x @ Wv
scores = Q @ K.T / (d_model ** 0.5)
weights = F.softmax(scores, dim=-1)
out = weights @ V
print("attention weights shape:", weights.shape)
print("output shape:", out.shape)
```

Note: using the same random `Wq, Wk, Wv` in the lab is for simplicity. Real models initialize them differently.

```python
# Which position does each query attend to most?
top_keys = weights.argmax(dim=-1)
for q_pos in range(seq_len):
    print(f"query {q_pos} -> key {top_keys[q_pos].item()}")
```

```python
class SelfAttention(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.Wq = torch.nn.Linear(d_model, d_model, bias=False)
        self.Wk = torch.nn.Linear(d_model, d_model, bias=False)
        self.Wv = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        d_k = Q.shape[-1]
        scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
        weights = F.softmax(scores, dim=-1)
        return weights @ V, weights

attn = SelfAttention(d_model=8)
x = torch.randn(4, 8)
out, w = attn(x)
print(out.shape, w.shape)
```

```python
import numpy as np

# NumPy: interpret attention as similarity matrix of projected tokens
rng = np.random.default_rng(0)
X = rng.standard_normal((5, 8))
Wq = rng.standard_normal((8, 8))
Wk = rng.standard_normal((8, 8))
Q, K = X @ Wq, X @ Wk
S = Q @ K.T / np.sqrt(8)
A = np.exp(S - S.max(axis=1, keepdims=True))
A /= A.sum(axis=1, keepdims=True)
print("self-attention weights shape:", A.shape)
```

```python
# Batch + sequence: (batch, seq, d_model)
B, n, d = 2, 6, 16
x = torch.randn(B, n, d)
layer = SelfAttention(d)
out, w = layer(x)  # need to adjust for batch — use batched matmul
# Batched scores: (B, n, n)
Q = x @ torch.randn(d, d)
K = x @ torch.randn(d, d)
V = x @ torch.randn(d, d)
scores = Q @ K.transpose(-2, -1) / (d ** 0.5)
weights = F.softmax(scores, dim=-1)
out = weights @ V
print(weights.shape)  # (2, 6, 6)
```

---

## 5. Visualizations

Self-attention weights reveal **linguistic or structural patterns** when trained. Even with random weights, you can visualize the mechanism.

```python
import numpy as np
import matplotlib.pyplot as plt

seq_len = 8
d_model = 16
rng = np.random.default_rng(42)
X = rng.standard_normal((seq_len, d_model))
Wq = rng.standard_normal((d_model, d_model))
Wk = rng.standard_normal((d_model, d_model))
Q, K = X @ Wq, X @ Wk
S = Q @ K.T / np.sqrt(d_model)
A = np.exp(S - S.max(axis=1, keepdims=True))
A /= A.sum(axis=1, keepdims=True)

tokens = [f"t{i}" for i in range(seq_len)]
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(A, cmap="Purples")
ax.set_xticks(range(seq_len), tokens)
ax.set_yticks(range(seq_len), tokens)
ax.set_xlabel("Key token")
ax.set_ylabel("Query token")
ax.set_title("Self-attention weights (random init)")
plt.colorbar(im)
plt.tight_layout()
plt.show()
```

**Reading the plot:** Row \(i\) shows how token \(i\) distributes attention across all tokens. Diagonal brightness often appears — tokens attend to themselves. Trained models show off-diagonal structure (syntax, coreference).

```python
# Synthetic: token 0 embeds similar to token 3
X = np.zeros((6, 4))
X[0] = [1, 0, 0, 0]
X[3] = [0.9, 0.1, 0, 0]
X[1:] = np.random.randn(5, 4) * 0.1
X[1] = [0, 1, 0, 0]
# ... simplified demo
Wq = Wk = np.eye(4)
Q, K = X @ Wq, X @ Wk
S = Q @ K.T
A = np.exp(S - S.max(axis=1, keepdims=True))
A /= A.sum(axis=1, keepdims=True)
plt.imshow(A, cmap="Blues")
plt.title("Similar tokens attract attention")
plt.xlabel("key")
plt.ylabel("query")
plt.colorbar()
plt.show()
```

---

## 6. Worked Examples

### Example 1: Shape trace

\(n = 4\), \(d_{model} = 8\).

- \(\mathbf{X}\): \((4, 8)\)
- \(\mathbf{Q}, \mathbf{K}, \mathbf{V}\): each \((4, 8)\)
- Scores \(\mathbf{Q}\mathbf{K}^\top\): \((4, 4)\)
- Weights \(\mathbf{A}\): \((4, 4)\)
- Output \(\mathbf{A}\mathbf{V}\): \((4, 8)\)

Each token outputs an 8-D vector that mixes information from all 4 tokens.

### Example 2: Self-attention row interpretation

If row 2 of \(\mathbf{A}\) is `[0.05, 0.70, 0.15, 0.10]`, token 2's new representation is:

\[
\mathbf{o}_2 = 0.05 \mathbf{v}_0 + 0.70 \mathbf{v}_1 + 0.15 \mathbf{v}_2 + 0.10 \mathbf{v}_3
\]

Token 2 pulls mostly from token 1's value.

### Example 3: Identity projections (thought experiment)

If \(\mathbf{W}_Q = \mathbf{W}_K = \mathbf{I}\), then scores are \(\mathbf{X}\mathbf{X}^\top\) — raw embedding similarities. Values would still need \(\mathbf{W}_V\) for nontrivial mixing.

### Example 4: Lab walkthrough

```python
import torch
import torch.nn.functional as F

seq_len, d_model = 4, 8
torch.manual_seed(0)
x = torch.randn(seq_len, d_model)
Wq = torch.randn(d_model, d_model)
Wk = torch.randn(d_model, d_model)
Wv = torch.randn(d_model, d_model)

Q, K, V = x @ Wq, x @ Wk, x @ Wv
weights = F.softmax(Q @ K.T / (d_model ** 0.5), dim=-1)
for i in range(seq_len):
    j = weights[i].argmax().item()
    print(f"position {i} attends most to position {j} (weight {weights[i,j]:.3f})")
```

### Example 5: Parameter count for projections

Three matrices \(\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V\) each have size \(d_{model} \times d_{model}\). Total projection parameters: \(3 d_{model}^2\) (ignoring bias). For \(d_{model} = 4096\), that is roughly 50 million parameters **per attention layer** from QKV alone — before output projection and MLP. Scale explains why LLM training demands significant compute and memory.

---

## 7. AI Connection

> 🧠 AI Insight
>
> GPT processes up to 128k+ tokens. Each layer's self-attention is an \(n \times n\) interaction matrix — expensive for long sequences (\(O(n^2)\) memory and compute). Research on linear attention, sliding windows, and state-space models targets this bottleneck. But the **idea** remains: every token gathers context from the sequence via learned similarity.

**Contextual embeddings:** Static word vectors (Word2Vec) are the same for every occurrence. Self-attention output for "bank" differs depending on neighboring tokens — **contextualized** representations.

**Positional information:** Pure self-attention is permutation-invariant. Transformers add **positional encodings** to \(\mathbf{X}\) so order matters. Without them, shuffling tokens would not change the output.

**BERT vs GPT:** BERT uses bidirectional self-attention (all positions visible). GPT uses **masked** self-attention (positions see only past tokens) — covered in the decoder chapter.

**Training:** Gradients flow through \(\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V\) via backpropagation. Attention weights are not stored parameters — they are recomputed each forward pass from current embeddings.

---

## 8. Common Mistakes

> ⚠️ Common Mistake
>
> **Assuming self-attention preserves input unchanged.** Output is a **mixture** of value vectors. Even with high self-weight, other tokens contribute unless attention is exactly one-hot on self.

> ⚠️ Common Mistake
>
> **Ignoring positional encoding.** Raw self-attention treats the input as a **set**, not a sequence. Production models always add position information before attention.

> ⚠️ Common Mistake
>
> **Using `d_model` vs `d_k` inconsistently.** In single-head attention, scaling uses the dimension of queries/keys. When heads split dimensions (next chapter), scale uses **per-head** \(d_k = d_{model} / h\).

**Correct understanding:** Self-attention = project X to Q,K,V, then scaled dot-product attention. Output replaces each token embedding with a context-aware blend.

---

## 9. Exercises

### Easy

1. For `seq_len=4`, `d_model=8`, list shapes of X, Q, K, V, weights, and output.
2. Run the lab notebook and print `argmax` key per query position.
3. Why are Wq, Wk, Wv separate matrices instead of one?

### Medium

4. Implement self-attention as a single `nn.Module` with three `nn.Linear` layers.
5. Add a residual connection: `x + self_attn(x)`. What shape constraint is required?
6. Plot attention heatmap for `seq_len=10` with random X.

### Hard

7. Explain permutation invariance: if you permute rows of X and permute outputs the same way, does untrained self-attention output permute identically? (Assume shared Wq=Wk=Wv=I.)
8. Count multiply-adds for self-attention with \(n=1024\), \(d=768\).
9. How would you add a padding mask so padded tokens receive zero attention?

### Challenge

10. **Context swap:** Build two sequences that differ only in token order. Show (with identity projections) that self-attention output is identical without positional encoding — demonstrating the need for position signals.

---

## 10. Mini Project

### Self-Attention Pattern Finder

1. Create 6 token embeddings where tokens 0 and 5 are nearly identical.
2. Run self-attention with `Wq = Wk = I`, `Wv = I`.
3. Plot attention heatmap — tokens 0 and 5 should attend to each other.
4. Add positional encoding (sinusoidal or learned) and observe diagonal shift.

```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

n, d = 6, 8
x = torch.randn(n, d) * 0.1
x[0] = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.])
x[5] = torch.tensor([0.95, 0.05, 0., 0., 0., 0., 0., 0.])

Wq = Wk = Wv = torch.eye(d)
Q, K, V = x @ Wq, x @ Wk, x @ Wv
w = F.softmax(Q @ K.T / (d ** 0.5), dim=-1)
out = w @ V

plt.imshow(w.detach().numpy(), cmap="Blues")
plt.xlabel("key position")
plt.ylabel("query position")
plt.title("Similar tokens (0 and 5) cross-attend")
plt.colorbar()
plt.show()
```

<details>
<summary>Mini project checklist</summary>

- [ ] Constructed similar token embeddings
- [ ] Heatmap shows cross-attention between 0 and 5
- [ ] Written explanation of why

</details>

---

## 11. Interview Questions

**Q1:** What is self-attention?

**A1:** Self-attention computes attention where queries, keys, and values all come from the same sequence. Each token is projected by learned matrices Wq, Wk, Wv, then attends over all tokens to produce a new context-aware representation. It captures long-range dependencies in one parallel operation.

**Q2:** Why do transformers need positional encodings with self-attention?

**A2:** Attention scores depend on content similarity, not position. Permuting tokens permutes outputs correspondingly — the operation is permutation-equivariant. Positional encodings inject order information into X before attention so "cat chased dog" differs from "dog chased cat."

**Q3:** What is the computational complexity of self-attention?

**A3:** For sequence length n and dimension d, computing QKᵀ is O(n²d) and multiplying weights by V is O(n²d). Memory for the attention matrix is O(n²). This quadratic cost in n is the main scalability challenge for very long contexts.

**Q4:** How does self-attention differ from a convolution or RNN?

**A4:** RNNs process sequentially — hard to parallelize, long paths for distant tokens. Convolutions have fixed receptive fields unless stacked deeply. Self-attention connects every pair of positions in one layer with path length O(1) for information mixing, fully parallelizable across positions.

---

## 12. Summary

### Key formulas

| Step | Formula |
|------|---------|
| Projections | \(\mathbf{Q} = \mathbf{X}\mathbf{W}_Q\), same for K, V |
| Self-attention | \(\text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_{model}}}\right)\mathbf{V}\) |
| Residual (typical) | \(\mathbf{X} + \text{SelfAttention}(\mathbf{X})\) |

### Key terminology

- **Self-attention** — Q, K, V derived from the same sequence
- **Projection matrices** — learned linear maps Wq, Wk, Wv
- **Contextual embedding** — representation after mixing other tokens
- **Permutation equivariance** — reorder inputs → reorder outputs (without position)
- **Positional encoding** — injects token order into embeddings

---

## 13. Preview

One self-attention head learns **one** way to relate tokens. Transformers run **multiple heads in parallel** — each with its own projections and its own \(d_k = d_{model} / h\) — then concatenate results. **Multi-head attention** lets the model attend to syntactic, semantic, and positional patterns simultaneously.

One head asks one kind of question. Many heads ask many questions at once.

**Next chapter:** [Multi-Head Attention](03-multi-head-attention.md)

---

## Lab

Companion notebook: [`app/transformers/02_self_attention.ipynb`](../../app/transformers/02_self_attention.ipynb)
