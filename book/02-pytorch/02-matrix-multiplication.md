# Matrix Multiplication

## 1. Introduction

If creating tensors is step zero, **matrix multiplication** is step one of every neural network forward pass. A linear layer computes `y = x @ W.T + b`.

> 📌 Preview — optional for now
>
> **Term:** attention (`Q @ K.T`)
> **One line:** compares query vectors to key vectors via dot products
> **Learn properly in:** [Attention Mechanism](../04-transformers/01-attention-mechanism.md)
> You can skip the details and keep reading.

> 📌 Preview — optional for now
>
> **Term:** backpropagation
> **One line:** applies the chain rule to compute gradients through composed layers
> **Learn properly in:** [Backpropagation](../03-neural-networks/03-backpropagation.md)
> You can skip the details and keep reading.

You learned matrices in the math module: rows and columns, dot products, the rule that columns of the first matrix must match rows of the second. PyTorch expresses all of this with `@` (or `torch.matmul`). This chapter makes that notation automatic — including the **batch** case where you multiply stacks of matrices at once.

After this chapter you will be able to:

- Multiply 2D matrices by hand and verify in PyTorch.
- Read shape annotations like `(m, n) @ (n, p) → (m, p)`.
- Perform batch matrix multiplication for neural network layers.
- Implement a linear layer from scratch using `@`.

**Where this appears in AI:** Every `nn.Linear` layer is matrix multiplication. Convolution can be rewritten as matrix multiplication. Optimization updates weights with element-wise ops, but forward and backward passes are dominated by matmul.

If nested loops over rows and columns feel familiar, you already understand the *pattern*: combine one row with one column at a time, accumulate a sum. Matrix multiplication is that pattern vectorized — thousands of dot products in one GPU call. Shape discipline at this step prevents most downstream training bugs.

**Suggested pacing (3 sessions):**

- Session A: §1–§3 + [cheatsheet](02-matrix-multiplication-cheatsheet.md) skim
- Session B: §4–§6 + lab notebook
- Session C: Easy–Medium exercises + readiness checks in §12

---

## 2. Intuition

> 💡 Intuition
>
> Matrix multiplication combines **rows of the first matrix** with **columns of the second** via dot products. If matrix \(A\) is "recipes" (each row lists ingredient amounts) and matrix \(B\) is "ingredient prices" (each column lists prices for one product), then \(A \times B\) gives the **cost of each recipe for each product** — each output cell is one dot product.

```
  A (2×3)          B (3×2)              A @ B (2×2)

  [1  2  3]        [1  4]               [1×1+2×2+3×3  1×4+2×5+3×6]
  [4  5  6]   @    [2  5]         =     [4×1+5×2+6×3  4×4+5×5+6×6]
                   [3  6]               [14  32]
                                        [32  77]
```

Each output element \((i, j)\) is the dot product of row \(i\) of \(A\) with column \(j\) of \(B\).

In a neural network, an input vector \(x\) of length \(n\) times a weight matrix \(W\) of shape \((m, n)\) produces an output vector of length \(m\). Each output neuron sums weighted inputs — exactly one dot product per row.

> 🔬 Deep Dive
>
> PyTorch supports three levels: 1D dot product (`(n,) @ (n,) → scalar`), 2D matrix multiply (`(m,n) @ (n,p) → (m,p)`), and batched multiply (`(b,m,n) @ (b,n,p) → (b,m,p)` or `(b,m,n) @ (n,p) → (b,m,p)` with broadcasting). Neural networks almost always use the batched form because you process many examples in parallel.

---

## 3. Formal Definitions

### Matrix multiplication

Given \(A \in \mathbb{R}^{m \times n}\) and \(B \in \mathbb{R}^{n \times p}\), the product \(C = AB\) has shape \(m \times p\).

**2×2 numeric example** (before the general formula):

\[
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad
B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}
\]

| Output cell | Calculation | Value |
|-------------|-------------|-------|
| \(C_{00}\) (row 0 · col 0) | \(1 \cdot 5 + 2 \cdot 7\) | 19 |
| \(C_{01}\) (row 0 · col 1) | \(1 \cdot 6 + 2 \cdot 8\) | 22 |
| \(C_{10}\) (row 1 · col 0) | \(3 \cdot 5 + 4 \cdot 7\) | 43 |
| \(C_{11}\) (row 1 · col 1) | \(3 \cdot 6 + 4 \cdot 8\) | 50 |

Each entry is one dot product: row \(i\) of \(A\) with column \(j\) of \(B\). The general rule:

\[
C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}
\]

> **Plain English**
> To fill cell \((i,j)\), multiply matching entries from row \(i\) of \(A\) and column \(j\) of \(B\), then add the products.

> **Python**
> `C[i, j] = (A[i, :] * B[:, j]).sum()`

- \(A_{ik}\) — element in row \(i\), column \(k\) of \(A\)
- \(B_{kj}\) — element in row \(k\), column \(j\) of \(B\)
- The sum runs over the shared inner dimension \(k = 1, \ldots, n\)

**Shape rule:** Inner dimensions must match. \((m, \color{red}{n}) @ (\color{red}{n}, p) \to (m, p)\).

### Linear layer (neural network)

\[
y = x W^\top + b
\]

> **Plain English**
> For each output neuron, take a weighted sum of all inputs (using one row of \(W\)), then add that neuron's bias.

> **Python**
> `y = x @ W.T + b`

- \(x\) — input row vector, shape \((1, n)\) or batch \((B, n)\)
- \(W\) — weights, shape \((m, n)\)
- \(b\) — bias, shape \((m,)\)
- \(y\) — output, shape \((B, m)\)

PyTorch's `nn.Linear(n, m)` stores \(W\) as `(m, n)` and computes `x @ W.T + b`.

---

## 4. Programming Perspective

| Math | PyTorch |
|------|---------|
| \(C = AB\) | `C = A @ B` or `torch.matmul(A, B)` |
| Element-wise multiply | `A * B` (not matrix multiply!) |
| Batch matmul | `(B, m, n) @ (B, n, p)` |
| Vector dot product | `(n,) @ (n,)` → scalar |

```python
import torch

A = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])   # shape (2, 3)
B = torch.tensor([[1.],
                  [2.],
                  [3.]])           # shape (3, 1)

C = A @ B
print("A @ B =\n", C)
# tensor([[14.],
#         [32.]])
print("shape:", C.shape)  # torch.Size([2, 1])
```

**Critical:** `@` is matrix multiply; `*` is element-wise multiply. Confusing them is a top beginner bug.

```python
# Element-wise — shapes must match exactly (or broadcast)
D = A * torch.tensor([[1., 1., 1.],
                      [2., 2., 2.]])
print("A * mask =\n", D)  # not the same as A @ B!
```

### Linear layer from scratch

```python
def linear(x, W, b):
    """x: (batch, in_features), W: (out_features, in_features), b: (out_features,)"""
    return x @ W.T + b

x = torch.randn(4, 3)      # batch of 4, 3 features
W = torch.randn(2, 3)      # 2 outputs, 3 inputs
b = torch.zeros(2)

y = linear(x, W, b)
print(y.shape)  # torch.Size([4, 2])
```

This is exactly what `nn.Linear(3, 2)` does internally.

---

## 5. Visualizations

Matrix multiplication is algebraic, but we can visualize the shapes and a simple 2D case.

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

A = torch.tensor([[1., 2.],
                  [3., 4.]])
B = torch.tensor([[0.5, 1.0],
                  [1.5, 0.0]])
C = A @ B

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
titles = ["A (2×2)", "B (2×2)", "A @ B (2×2)"]
tensors = [A, B, C]

for ax, t, title in zip(axes, tensors, titles):
    im = ax.imshow(t.numpy(), cmap="viridis", aspect="auto", vmin=0, vmax=4)
    ax.set_title(title)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            ax.text(j, i, f"{t[i,j].item():.1f}", ha="center", va="center", color="white")
    ax.set_xlabel("column")
    ax.set_ylabel("row")

plt.tight_layout()
plt.show()
```

**How to read:** Each heatmap shows one matrix. The third panel is the result — each cell is a dot product of one row of \(A\) with one column of \(B\). For example, top-left of \(C\): \(1 \times 0.5 + 2 \times 1.5 = 3.5\).

```python
# Visualize batch matmul: 3 separate matrix multiplies in parallel
batch_A = torch.randn(3, 2, 4)   # 3 matrices of shape (2, 4)
batch_B = torch.randn(3, 4, 5)   # 3 matrices of shape (4, 5)
batch_C = batch_A @ batch_B        # (3, 2, 5)

print("batch_A:", batch_A.shape)
print("batch_B:", batch_B.shape)
print("batch_C:", batch_C.shape)
# Each of the 3 "slices" along dim 0 is an independent matmul
```

---

## 6. Worked Examples

### Example 1: Basic 2D multiply

\(A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}\), \(B = \begin{bmatrix} 5 \\ 6 \end{bmatrix}\) (shape \(2 \times 1\))

\[
A @ B = \begin{bmatrix} 1 \cdot 5 + 2 \cdot 6 \\ 3 \cdot 5 + 4 \cdot 6 \end{bmatrix} = \begin{bmatrix} 17 \\ 39 \end{bmatrix}
\]

```python
A = torch.tensor([[1., 2.], [3., 4.]])
B = torch.tensor([[5.], [6.]])
print(A @ B)  # tensor([[17.], [39.]])
```

### Example 2: Shape mismatch error

```python
X = torch.randn(3, 4)
W = torch.randn(5, 4)  # wrong: inner dims 4 and 4 OK, but...
Y = torch.randn(3, 5)
# X @ W.T works: (3,4) @ (4,5) -> (3,5)
print((X @ W.T).shape)  # torch.Size([3, 5])
```

Always trace shapes: `(batch, in) @ (in, out).T` or `(batch, in) @ (out, in).T` depending on weight layout.

### Example 3: Batch matmul for batched dot products

Query \(Q\): `(2, 3, 4)` — 2 stacks, 3 tokens, 4 dims. Key \(K\): `(2, 3, 4)`.

Scores: `(2, 3, 4) @ (2, 4, 3)` if we transpose keys — simplified 2-stack case:

```python
Q = torch.randn(2, 3, 8)   # 2 heads, 3 tokens, 8 dims
K = torch.randn(2, 3, 8)
scores = Q @ K.transpose(-2, -1)  # (2, 3, 3) — each token vs each token
print(scores.shape)  # torch.Size([2, 3, 3])
```

This batched dot-product pattern is the mechanical core of scaled dot-product attention (covered in the transformers module).

### Example 4: Full mini forward pass

```python
batch, in_f, hidden, out_f = 8, 784, 128, 10

x = torch.randn(batch, in_f)
W1 = torch.randn(hidden, in_f)
b1 = torch.zeros(hidden)
W2 = torch.randn(out_f, hidden)
b2 = torch.zeros(out_f)

h = torch.relu(x @ W1.T + b1)   # hidden layer
logits = h @ W2.T + b2           # output layer
print(logits.shape)  # torch.Size([8, 10]) — 8 images, 10 classes
```

Two matrix multiplies, one ReLU — a tiny MLP.

---

## 7. AI Connection

> 🧠 AI Insight
>
> A transformer block with hidden size 768 and FFN inner size 3072 performs matmuls of shapes roughly `(batch, seq, 768) @ (768, 3072)` and back. At scale, these operations dominate compute time — which is why GPUs exist and why FlashAttention optimizes the `Q @ K.T` multiply.

**Linear layers:** `output = input @ weight.T + bias` — one matmul per layer.

**Attention:** `Attention(Q,K,V) = softmax(Q @ K.T / sqrt(d)) @ V` — two matmuls per head.

**Embeddings:** Lookup is indexing, but the embedding matrix is `(vocab_size, embed_dim)` and often multiplied in projection layers.

**Backpropagation:** Gradients flow backward through matmuls. If `Y = X @ W`, then \(\partial L / \partial W = X.T @ \partial L / \partial Y\). PyTorch autograd handles this; understanding forward shapes helps you debug backward shape errors.

**Optimization:** Weight updates use element-wise ops, but each training step's forward and backward passes are matmul-heavy. Mixed-precision training (`float16` matmul) exists because of this dominance.

---

## 8. Common Mistakes

> ⚠️ Common Mistake
>
> **Using `*` instead of `@`.** `A * B` multiplies element-by-element. `A @ B` is matrix multiply. For neural layers, you almost always want `@`.

> ⚠️ Common Mistake
>
> **Transposing the wrong matrix.** `nn.Linear` stores weights as `(out_features, in_features)`. You need `x @ W.T`, not `x @ W`, when `x` is `(batch, in_features)`.

> ⚠️ Common Mistake
>
> **Ignoring batch dimension.** `(784,) @ (256, 784)` fails or behaves unexpectedly. Inputs should be `(batch, 784)`. Always include the batch axis even when batch size is 1: `(1, 784)`.

> ⚠️ Common Mistake
>
> **Assuming matmul is commutative.** \(AB \neq BA\) in general. Order matters. `A @ B` and `B @ A` give different results and often different shapes.

**Correct understanding:** Trace shapes on paper before coding. Use `@` for linear layers and attention. Remember weight layout in `nn.Linear`.

---

## 9. Exercises

### Easy

1. Compute by hand: `(2×3) @ (3×1)` with \(A = [[1,0,2],[3,1,0]]\), \(B = [[1],[2],[3]]\). Verify in PyTorch.
2. Create random `A` (3×4) and `B` (4×2). Print shapes of `A`, `B`, and `A @ B`.
3. What is the shape of `(5, 10) @ (10, 3)`?
4. Implement dot product of vectors `u` and `v` (length 5) using `@` and using `(u * v).sum()`. Compare results.

### Medium

5. Implement `nn.Linear(10, 5)` forward pass using only `@` and `+`, no `nn` modules.
6. Batch matmul: create `(4, 3, 5) @ (4, 5, 2)`. Print output shape. Explain what the `4` means.
7. Given `x` of shape `(32, 512)` and `W` of shape `(256, 512)`, compute output of shape `(32, 256)`.
8. Show that `(A @ B).T == B.T @ A.T` with random 2×3 and 3×4 matrices.

### Hard

9. Why does `torch.matmul` broadcast `(B, n, m) @ (m, p)` to `(B, n, p)`? Create an example.
10. Count total multiply-adds in `x @ W.T` for `x` shape `(B, d_in)`, `W` shape `(d_out, d_in)`.

### Challenge

11. **MLP from scratch:** Two layers, ReLU, no `nn.Linear` — only `@`, `+`, `torch.relu`. Input `(64, 784)`, hidden 128, output 10.

> 📌 Preview — optional for now
>
> **Term:** attention scores (`Q @ K.T`)
> **One line:** token–token similarity matrix before softmax
> **Learn properly in:** [Self-Attention](../04-transformers/02-self-attention.md)
> You can skip the details and keep reading.

12. Simulate attention scores: `Q`, `K` shape `(1, 8, 64)` (1 head, 8 tokens, 64 dims). Compute `(1, 8, 8)` score matrix.
13. **Matmul profiler:** Time `(1000, 1000) @ (1000, 1000)` on CPU vs GPU. Report elapsed time and GFLOPS estimate.

---

## 10. Mini Project

### Linear Layer Laboratory

Implement and compare three ways to apply a linear transformation:

1. Manual `@ W.T + b`
2. `torch.nn.functional.linear(x, W, b)`
3. `nn.Linear` module

Verify all three produce identical outputs on the same random input.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
in_f, out_f, batch = 16, 8, 32

x = torch.randn(batch, in_f)
W = torch.randn(out_f, in_f)
b = torch.randn(out_f)

y1 = x @ W.T + b
y2 = F.linear(x, W, b)
layer = nn.Linear(in_f, out_f)
layer.weight.data = W
layer.bias.data = b
y3 = layer(x)

print("manual vs F.linear:", torch.allclose(y1, y2))
print("manual vs nn.Linear:", torch.allclose(y1, y3))
print("output shape:", y1.shape)
```

<details>
<summary>Mini project checklist</summary>

- [ ] Three implementations match numerically
- [ ] Shapes documented at each step
- [ ] Brief comment on when to use each API

</details>

---

## 11. Interview Questions

**Q1:** What are the shape requirements for matrix multiplication?

**A1:** For `A @ B`, the number of columns in `A` must equal the number of rows in `B`. If `A` is `(m, n)` and `B` is `(n, p)`, the result is `(m, p)`. The inner dimension `n` is contracted (summed over). Mismatched inner dimensions raise a runtime error in PyTorch.

**Q2:** What is the difference between `A @ B` and `A * B` in PyTorch?

**A2:** `A @ B` is matrix multiplication (dot products of rows and columns). `A * B` is element-wise (Hadamard) multiplication — each position multiplied independently. Shapes for `*` must match or broadcast; shapes for `@` follow the `(m,n) @ (n,p)` rule. Neural network layers use `@`.

**Q3:** How does batch matrix multiplication work?

**A3:** If `A` is `(B, m, n)` and `B` is `(B, n, p)`, PyTorch performs `B` independent matrix multiplies in parallel, producing `(B, m, p)`. The batch dimension is leading. This lets you process an entire mini-batch of examples in one GPU kernel — essential for training speed.

**Q4:** In `nn.Linear(in_features, out_features)`, what is the shape of the weight matrix?

**A4:** Weight shape is `(out_features, in_features)`. For input `x` of shape `(batch, in_features)`, the forward pass computes `x @ weight.T + bias`, yielding `(batch, out_features)`. Storing weights as `(out, in)` matches the convention that each row is one neuron's weights across all inputs.

**Q5:** Where does matrix multiplication appear in transformer attention?

**A5:** Scaled dot-product attention computes `scores = Q @ K.transpose(-2, -1)` for token-token similarity, then `output = softmax(scores) @ V` to combine values. Both steps are batched matmuls over heads and sequences. Multi-head attention stacks several such matmuls in parallel.

Understanding matmul complexity helps you reason about cost: multiplying `(m,n) @ (n,p)` requires roughly \(2mnp\) multiply-add operations — dominant in large language model inference and training.

---

## 12. Summary

### Core takeaways (must know)

- `@` is matrix multiply; `*` is element-wise — never confuse them
- Shape rule: `(m, n) @ (n, p) → (m, p)`; inner dimensions must match
- `nn.Linear` computes `x @ W.T + b` with weights stored as `(out, in)`
- Batch dimension leads: `(B, m, n) @ (B, n, p) → (B, m, p)`

### Preview terms (optional until later)

- Attention (`Q @ K.T`), backprop through matmul, transformer blocks — see [Vocabulary Roadmap](../00-intro/04-vocabulary-roadmap.md)

### Key formulas

| Operation | Shapes | Result |
|-----------|--------|--------|
| Matrix multiply | `(m, n) @ (n, p)` | `(m, p)` |
| Linear layer | `(B, d_in) @ (d_out, d_in).T + b` | `(B, d_out)` |
| Dot product | `(n,) @ (n,)` | scalar |
| Batch matmul | `(B, m, n) @ (B, n, p)` | `(B, m, p)` |

\[
C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}
\]

### Key terminology

- **Matrix multiplication** — bilinear operation combining rows and columns via dot products
- **Inner dimension** — shared axis that must match for `@`
- **Batch dimension** — leading axis for parallel examples
- **Broadcasting** — PyTorch rules for matmul with mismatched batch ranks
- **FLOPs** — floating-point ops; matmul dominates NN compute

### Readiness checks

Before the next chapter, you should be able to:

1. Multiply a `(2, 3)` matrix by a `(3, 1)` column vector by hand and verify with PyTorch.
2. Trace shapes for `(32, 512) @ (256, 512).T` without running code.
3. Explain why `A * B` and `A @ B` give different results.
4. Implement `linear(x, W, b)` using only `@` and `+`.
5. State the inner-dimension rule for batch matmul `(B, m, n) @ (B, n, p)`.

If any item is shaky, reread §3 and the [cheatsheet](02-matrix-multiplication-cheatsheet.md).

---

## 13. Preview

Matrix multiplication assumes you have matrices oriented correctly. Often you need to **swap rows and columns** — transpose — or **rearrange dimensions** before multiplying. The next chapter covers transposing tensors, including the critical `K.T` in attention and weight layout in linear layers.

**Next chapter:** [Transposing Tensors](03-transposing-tensors.md)

---

## Lab

Companion notebook: [`app/pytorch/02_matrix_multiplication.ipynb`](../../app/pytorch/02_matrix_multiplication.ipynb)

## Review

- Cheatsheet: [Matrix Multiplication — Cheatsheet](02-matrix-multiplication-cheatsheet.md)
- Jargon: [Vocabulary Roadmap](../00-intro/04-vocabulary-roadmap.md)
