# Attention Mechanism

## 1. Introduction

> **Payoff chapter:** **Dot products** and **matrix multiply** from [Vectors](../01-math/03-vectors.md) and [Matrices](../01-math/05-matrices.md), and **softmax** from [Probability](../01-math/06-probability.md), come together here.

Before transformers, sequence models relied on recurrence — processing tokens one at a time, carrying hidden state forward. **Attention** offered a different idea: at each step, **look at everything** and decide what matters, using learned **queries**, **keys**, and **values**.

The core computation is elegant: compare a query to every key, turn comparisons into weights with **softmax**, and take a weighted sum of values. That is a differentiable, parallelizable routing mechanism — the foundation of GPT, BERT, and modern vision transformers.

After this chapter you will be able to:

- Explain what **Query (Q)**, **Key (K)**, and **Value (V)** represent.
- Compute attention scores, softmax weights, and output by hand on a 2×2 example.
- Implement scaled dot-product attention in PyTorch.
- Describe why we divide scores by \(\sqrt{d_k}\).

**Where this appears in AI:** Every transformer layer uses attention. Cross-attention in encoder-decoder models (translation) lets the decoder query encoder states. Self-attention (next chapter) is the special case where Q, K, V all come from the same sequence.

**Suggested pacing (3 sessions):**

- Session A: §1–§3 + [cheatsheet](01-attention-mechanism-cheatsheet.md) skim
- Session B: §4–§6 + lab notebook
- Session C: Easy–Medium exercises + readiness checks in §12

---

## 2. Intuition

> 💡 Intuition
>
> Imagine searching a library. Your **query** is the question you ask. Each book spine has a **key** — a short label describing its topic. You compare your question to each label (dot product = similarity). Books with matching topics get high weight. You then read a **blend** of the books' contents (**values**) weighted by relevance. Attention does exactly this with vectors instead of books.

```
  Query q  ──compare──►  Keys [k₁, k₂, ...]  ──softmax──►  weights [α₁, α₂, ...]
                                                                              │
  Values [v₁, v₂, ...] ◄──────────── weighted sum ◄──────────────────────────┘
                              output = α₁v₁ + α₂v₂ + ...
```

Each row of \(\mathbf{Q}\) is one query. Each row of \(\mathbf{K}\) is one key. Each row of \(\mathbf{V}\) is one value vector. One query attends over all keys and produces one output vector.

The separation between keys and values is subtle but important. A key is used for **selection**: it helps decide how relevant an item is. A value is used for **content transfer**: it is the information that gets blended into the output after relevance has been computed. In a search engine analogy, the title and metadata might help you find a document, while the document body contains the information you actually read. In attention, both are learned vectors, but they serve different roles in the computation.

This also explains why attention is more flexible than a fixed average. A normal average would mix all values equally. Attention creates a custom average for each query. Query 0 may mostly read value 0, query 1 may split between values 0 and 2, and query 2 may ignore value 0 completely. The model learns these routing patterns from data rather than from hand-written rules.

> 🔬 Deep Dive
>
> Dot product \(\mathbf{q} \cdot \mathbf{k}\) measures alignment: large when vectors point the same direction, small or negative when opposed. Softmax converts raw scores into a **probability distribution** over positions — weights sum to 1, all positive. The output is a convex combination of value vectors.

---

## 3. Formal Definitions

### Query, Key, Value matrices

For \(n\) positions (sequence length) and dimension \(d_k\) for queries/keys, \(d_v\) for values:

\[
\mathbf{Q} \in \mathbb{R}^{n \times d_k}, \quad
\mathbf{K} \in \mathbb{R}^{n \times d_k}, \quad
\mathbf{V} \in \mathbb{R}^{n \times d_v}
\]

Row \(i\) of \(\mathbf{Q}\) is query \(i\). Row \(j\) of \(\mathbf{K}\) is key \(j\). Row \(j\) of \(\mathbf{V}\) is value \(j\).

### Attention scores

\[
\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}
\]

\(\mathbf{S}\) has shape \((n \times n)\). Entry \(S_{ij} = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}}\) measures how much query \(i\) attends to key \(j\).

### Attention weights

Apply **softmax** row-wise:

\[
\mathbf{A} = \text{softmax}(\mathbf{S}, \text{dim}=-1)
\]

\[
A_{ij} = \frac{e^{S_{ij}}}{\sum_{j'} e^{S_{ij'}}}
\]

Each row of \(\mathbf{A}\) sums to 1.

### Output

\[
\mathbf{O} = \mathbf{A} \mathbf{V}
\]

Shape \((n \times d_v)\). Row \(i\) of \(\mathbf{O}\) is the weighted sum of all value rows, weights from row \(i\) of \(\mathbf{A}\).

### Scaling factor \(\sqrt{d_k}\)

Without scaling, dot products grow with dimension \(d_k\), pushing softmax into extreme values (near one-hot). Dividing by \(\sqrt{d_k}\) keeps score variance stable as dimension increases.

Here is the step-by-step reason. A dot product adds \(d_k\) multiplication results:

\[
\mathbf{q} \cdot \mathbf{k} = q_1k_1 + q_2k_2 + \cdots + q_{d_k}k_{d_k}
\]

If those terms are roughly centered around zero with similar variance, adding more terms makes the typical magnitude of the sum grow. Large positive and negative scores make softmax saturate: one entry becomes very close to 1 and the others become very close to 0. Saturation is a problem for optimization because tiny probabilities often produce tiny useful gradients for the alternatives. Dividing by \(\sqrt{d_k}\) keeps the score scale closer to what softmax can handle.

In transformer engineering, this small scaling factor matters because \(d_k\) can be 64, 128, or larger. Without it, deeper attention stacks become harder to train, and early random scores can become too sharp before the model has learned meaningful relationships.

| Symbol | Shape | Meaning |
|--------|-------|---------|
| \(\mathbf{Q}\) | \((n, d_k)\) | Queries |
| \(\mathbf{K}\) | \((n, d_k)\) | Keys |
| \(\mathbf{V}\) | \((n, d_v)\) | Values |
| \(\mathbf{S}\) | \((n, n)\) | Scaled scores |
| \(\mathbf{A}\) | \((n, n)\) | Attention weights |
| \(\mathbf{O}\) | \((n, d_v)\) | Output |

---

## 4. Programming Perspective

```python
import torch
import torch.nn.functional as F

Q = torch.tensor([[1., 0.], [0., 1.]])
K = torch.tensor([[1., 0.], [0., 1.]])
V = torch.tensor([[2., 3.], [4., 5.]])

d_k = Q.shape[-1]
scores = Q @ K.T / (d_k ** 0.5)
weights = F.softmax(scores, dim=-1)
out = weights @ V
print("weights:\n", weights)
print("output:\n", out)
```

**Identity Q and K:** Query 0 aligns with key 0; query 1 with key 1. Weights are nearly identity; output ≈ V.

```python
# Change Q so row 0 attends more to key 1
Q2 = torch.tensor([[0., 1.], [0., 1.]])
scores2 = Q2 @ K.T / (d_k ** 0.5)
weights2 = F.softmax(scores2, dim=-1)
out2 = weights2 @ V
print("row 0 weights:", weights2[0])
print("row 0 output:", out2[0])  # closer to V[1] = [4, 5]
```

```python
def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / (d_k ** 0.5)
    weights = F.softmax(scores, dim=-1)
    return weights @ V, weights

O, A = scaled_dot_product_attention(Q, K, V)
print("output shape:", O.shape)
print("weights sum per row:", A.sum(dim=-1))
```

```python
import numpy as np

# NumPy version for visualization workflows
def attention_numpy(Q, K, V):
    d_k = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d_k)
    S = S - S.max(axis=-1, keepdims=True)  # numerical stability
    A = np.exp(S)
    A = A / A.sum(axis=-1, keepdims=True)
    return A @ V, A
```

```python
# Batch dimension: (batch, n, d)
B, n, d = 2, 3, 4
Q_b = torch.randn(B, n, d)
K_b = torch.randn(B, n, d)
V_b = torch.randn(B, n, d)
scores_b = Q_b @ K_b.transpose(-2, -1) / (d ** 0.5)
weights_b = F.softmax(scores_b, dim=-1)
out_b = weights_b @ V_b
print(out_b.shape)  # (2, 3, 4)
```

---

## 5. Visualizations

Attention weights form an \(n \times n\) matrix — perfect for heatmaps. Rows are queries; columns are keys.

```python
import numpy as np
import matplotlib.pyplot as plt

Q = np.array([[1., 0.], [0.5, 0.5], [0., 1.]])
K = np.eye(2)
V = np.array([[1., 0.], [0., 1.]])

d_k = Q.shape[-1]
S = Q @ K.T / np.sqrt(d_k)
S = S - S.max(axis=1, keepdims=True)
A = np.exp(S)
A = A / A.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(A, cmap="Blues", vmin=0, vmax=1)
ax.set_xlabel("Key position")
ax.set_ylabel("Query position")
ax.set_title("Attention weight heatmap")
plt.colorbar(im, label="weight")
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center", color="black")
plt.tight_layout()
plt.show()
```

**Reading the heatmap:** Bright cell \((i, j)\) means query \(i\) strongly attends to key \(j\). Each row must sum to 1 — a probability distribution over keys.

```python
# Effect of scaling: softmax sharpness vs d_k
dims = [4, 16, 64, 256]
q = np.random.randn(1, max(dims))
k = np.random.randn(4, max(dims))

fig, axes = plt.subplots(1, len(dims), figsize=(14, 3))
for ax, d in zip(axes, dims):
    S = (q[:, :d] @ k[:, :d].T) / np.sqrt(d)
    A = np.exp(S - S.max())
    A = A / A.sum()
    ax.bar(range(4), A[0])
    ax.set_title(f"d_k={d}")
    ax.set_ylim(0, 1)
plt.suptitle("Without scaling, larger d_k → sharper softmax")
plt.tight_layout()
plt.show()
```

---

## 6. Worked Examples

### Example 1: 2×2 by hand

\(\mathbf{Q} = \mathbf{K} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}\), \(d_k = 2\), \(\mathbf{V} = \begin{bmatrix} 2 & 3 \\ 4 & 5 \end{bmatrix}\).

**Scores:** \(\mathbf{Q}\mathbf{K}^\top = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}\). Divide by \(\sqrt{2}\):

\[
\mathbf{S} = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
\]

**Softmax row 0:** \(e^{1/\sqrt{2}} \approx 2.19\), \(e^0 = 1\). Sum ≈ 3.19.

\(A_{00} \approx 0.69\), \(A_{01} \approx 0.31\).

**Output row 0:** \(0.69 \times [2,3] + 0.31 \times [4,5] \approx [2.6, 3.6]\).

### Example 2: Uniform attention

If all scores in a row are equal, softmax gives uniform weights \(1/n\). Output is the **average** of all value vectors.

### Example 3: One-hot attention

If one score dominates, weights approach \([0, \ldots, 1, \ldots, 0]\). Output copies one value row — **hard selection** (softly, via softmax).

### Example 4: PyTorch verification

```python
import torch
import torch.nn.functional as F

Q = torch.tensor([[1., 0.], [0., 1.]])
K = Q.clone()
V = torch.tensor([[2., 3.], [4., 5.]])
d_k = 2.0
S = Q @ K.T / (d_k ** 0.5)
A = F.softmax(S, dim=-1)
O = A @ V
print("A[0]:", A[0].tolist())
print("O[0]:", O[0].tolist())
```

### Example 5: Step-by-step softmax on one row

Suppose one query's scores (after scaling) are \([2.0, 1.0, 0.5]\).

**Step 1:** Exponentiate: \(e^2 \approx 7.39\), \(e^1 \approx 2.72\), \(e^{0.5} \approx 1.65\).

**Step 2:** Sum: \(7.39 + 2.72 + 1.65 \approx 11.76\).

**Step 3:** Normalize: weights \(\approx [0.63, 0.23, 0.14]\). They sum to 1.

**Step 4:** If values are \(\mathbf{v}_0 = [1,0]\), \(\mathbf{v}_1 = [0,1]\), \(\mathbf{v}_2 = [1,1]\), output is \(0.63[1,0] + 0.23[0,1] + 0.14[1,1] \approx [0.77, 0.37]\).

Softmax turns arbitrary scores into a **mixture coefficient** for each value vector. Training adjusts Q and K projections so the right values receive high weight for each task.

### Example 6: Attention as matrix multiplication chain

The full operation is three matrix multiplies: \(\mathbf{Q}\mathbf{K}^\top\) (scores), softmax (nonlinear, row-wise), \(\mathbf{A}\mathbf{V}\) (output). Every step is differentiable — backpropagation flows through Q, K, and V. This is why attention can be learned end-to-end with gradient descent, unlike hard nearest-neighbor lookup.

---

## 7. AI Connection

> 🧠 AI Insight
>
> Machine translation (Bahdanau et al., 2014) introduced attention so decoders could **look back** at relevant source words. Transformers (Vaswani et al., 2017) removed recurrence entirely — only attention and feed-forward layers. GPT's "next token prediction" is powered by stacked self-attention layers that route information between all previous tokens.

**Cross-attention:** Decoder queries attend to encoder keys/values. Used in original transformer encoder-decoder architecture.

**Retrieval:** Attention is soft dictionary lookup. Modern RAG systems retrieve documents externally; attention retrieves **within** the context window.

**Vision transformers (ViT):** Image patches become tokens; self-attention relates patches spatially.

**Differentiability:** Unlike hard indexing, softmax attention is smooth. Gradients flow to Q, K, V projections — learned during backpropagation.

---

## 8. Common Mistakes

> ⚠️ Common Mistake
>
> **Softmax on wrong dimension.** For scores shape `(n, n)`, softmax must be over **keys** (last dimension, `dim=-1`). Softmax over queries mixes unrelated positions.

> ⚠️ Common Mistake
>
> **Forgetting scale \(\sqrt{d_k}\).** Training becomes unstable for large \(d_k\); attention collapses to one-hot prematurely.

> ⚠️ Common Mistake
>
> **Confusing attention output shape with V.** Output has shape `(n, d_v)`, not `(n, d_k)`. If \(d_v \neq d_k\), dimensions differ by design.

**Correct understanding:** Scores = scaled QKᵀ. Weights = row softmax. Output = weights @ V.

---

## 9. Exercises

### Easy

1. If attention weights for one query are `[0.8, 0.2]` and values are `[1,0]` and `[0,1]`, what is the output?
2. Why do attention weights in each row sum to 1?
3. Run the lab notebook and print `weights` and `out`.

### Medium

4. Modify Q so query 0 attends 90%+ to key 1. Verify numerically.
5. What happens to output if all keys are identical?
6. Implement attention with `torch.einsum` instead of `@`.

### Hard

7. For \(d_k = 64\), if dot products have variance 64, what is variance after scaling?
8. Derive \(\frac{\partial L}{\partial \mathbf{Q}}\) given upstream gradient \(\frac{\partial L}{\partial \mathbf{O}}\) (conceptual outline).

### Challenge

9. Prove that softmax is invariant to adding a constant to all scores in a row. *(See also [Probability](../01-math/06-probability.md) §3 for the stability trick.)*

10. **Attention playground:** Generate random Q, K, V for \(n=8\), plot heatmap, animate how output changes as you rotate Q's first row toward different keys.

---

## 10. Mini Project

### Attention Inspector

Build a tool that:

1. Takes small Q, K, V tensors.
2. Prints scores, weights, output.
3. Plots heatmap of weights.
4. Highlights which key each query attends to most (`argmax`).

```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def inspect_attention(Q, K, V):
    d_k = Q.shape[-1]
    S = Q @ K.T / (d_k ** 0.5)
    A = F.softmax(S, dim=-1)
    O = A @ V
    print("Scores:\n", S)
    print("Weights:\n", A)
    print("Output:\n", O)
    print("Argmax keys per query:", A.argmax(dim=-1).tolist())
    plt.imshow(A.detach().numpy(), cmap="Blues")
    plt.xlabel("key")
    plt.ylabel("query")
    plt.title("Attention weights")
    plt.colorbar()
    plt.show()
    return O, A

Q = torch.tensor([[1., 0.], [0.3, 0.7], [0., 1.]])
K = torch.eye(2)
V = torch.tensor([[10., 0.], [0., 10.]])
inspect_attention(Q, K, V)
```

<details>
<summary>Mini project checklist</summary>

- [ ] Scores, weights, output printed
- [ ] Heatmap with labeled axes
- [ ] Argmax key per query reported

</details>

---

## 11. Interview Questions

**Q1:** Explain scaled dot-product attention in three sentences.

**A1:** Compute similarity between each query and every key via dot product, producing an \(n \times n\) score matrix. Divide scores by \(\sqrt{d_k}\) for stability, then apply row-wise softmax to get attention weights summing to 1. Multiply weights by value vectors to produce output — each position's output is a weighted blend of all values.

**Q2:** Why divide by \(\sqrt{d_k}\)?

**A2:** Dot products of random \(d_k\)-dimensional vectors have variance proportional to \(d_k\). Large scores drive softmax to extreme, near-one-hot distributions with tiny gradients. Scaling keeps score magnitudes stable as model dimension grows, improving trainability.

**Q3:** What is the difference between cross-attention and self-attention?

**A3:** Cross-attention uses queries from one sequence and keys/values from another (e.g., decoder attending to encoder). Self-attention uses the same sequence for Q, K, and V — each position attends to all positions in the same sequence. Self-attention is the topic of the next chapter.

**Q4:** What shapes are Q, K, V, and the output?

**A4:** Q and K: `(n, d_k)`. V: `(n, d_v)`. Scores/weights: `(n, n)`. Output: `(n, d_v)`. With batch: prepend batch dimension to all.

---

## 12. Summary

### Key formulas

| Step | Formula |
|------|---------|
| Scores | \(\mathbf{S} = \mathbf{Q}\mathbf{K}^\top / \sqrt{d_k}\) |
| Weights | \(\mathbf{A} = \text{softmax}(\mathbf{S})\) row-wise |
| Output | \(\mathbf{O} = \mathbf{A}\mathbf{V}\) |
| Combined | \(\text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}\) |

### Key terminology

- **Query** — what a position is looking for
- **Key** — what a position advertises about itself
- **Value** — content to aggregate if attended to
- **Scaled dot-product attention** — standard attention in transformers
- **Softmax** — converts scores to normalized weights

### Readiness checks

Before **Self-Attention**, you should be able to:

1. Compute scaled dot-product attention on a 2×2 example by hand.
2. Explain what Q, K, and V represent in plain language.
3. Implement attention with `Q @ K.T` and `softmax` in PyTorch.
4. State why scores are divided by \(\sqrt{d_k}\).
5. Trace shapes through Q, K, V for a small batch.

If any item is shaky, reread §6 and the [cheatsheet](01-attention-mechanism-cheatsheet.md).

---

## 13. Preview

This chapter used Q, K, V as **given** matrices. In transformers, they are **learned linear projections** of the same input sequence — that is **self-attention**. The next chapter derives \(\mathbf{Q} = \mathbf{X}\mathbf{W}_Q\), \(\mathbf{K} = \mathbf{X}\mathbf{W}_K\), \(\mathbf{V} = \mathbf{X}\mathbf{W}_V\) and shows how every token attends to every other token in one parallel pass.

Attention is the mechanism. Self-attention is how transformers apply it to language.

**Next chapter:** [Self-Attention](02-self-attention.md)

---

## Lab

Companion notebook: [`app/transformers/01_attention_mechanism.ipynb`](../../app/transformers/01_attention_mechanism.ipynb)

## Review

- Cheatsheet: [Attention Mechanism — Cheatsheet](01-attention-mechanism-cheatsheet.md)
- Jargon: [Vocabulary Roadmap](../../00-intro/04-vocabulary-roadmap.md)
