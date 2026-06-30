# Matrices

## 1. Introduction

Vectors hold lists of numbers. **Matrices** hold grids of numbers — rows and columns — and represent the most important operation in deep learning: **linear transformation**. When you write `nn.Linear(768, 3072)`, you are creating a weight matrix. When a transformer computes attention scores \(\mathbf{Q}\mathbf{K}^\top\), it is multiplying matrices.

If vectors are arrows, matrices are machines that take vectors in and spit vectors out — stretching, rotating, and mixing dimensions according to learned weights.

After this chapter you will be able to:

- Read matrix **shapes** and predict whether multiplication is valid.
- Multiply matrices by hand for small examples, and at scale with NumPy.
- Express a **linear layer** as \(y = Wx + b\).
- Understand why batching stacks vectors into matrices.
- Connect matrix multiplication to **attention** score computation.
- Avoid the most common shape errors in PyTorch.

**Where this appears in AI:** Every forward pass through a dense layer is matrix multiplication. Convolution can be expressed as matrix multiply. Attention forms an \((\text{seq}, \text{seq})\) matrix of scores via \(\mathbf{Q}\mathbf{K}^\top\). GPUs are optimized for large matrix multiplies — the reason deep learning became practical.

**Suggested pacing (3 sessions):**

- Session A: §1–§3 + [cheatsheet](05-matrices-cheatsheet.md) skim
- Session B: §4–§6 + lab notebook
- Session C: Easy–Medium exercises + readiness checks in §12

---

## 2. Intuition

> 💡 Intuition
>
> A matrix is a spreadsheet of numbers. Each **row** can be a vector. Multiplying a matrix by a vector computes many dot products at once — one per row of the matrix against the input vector.

```
  W (3×2 matrix)     x (2-vector)     =    y (3-vector)
  ┌         ┐       ┌   ┐              ┌   ┐
  │ row 0   │   ·   │   │    =         │ y0│  = row0 · x
  │ row 1   │       │ x │              │ y1│  = row1 · x
  │ row 2   │       │   │              │ y2│  = row2 · x
  └         ┘       └   ┘              └   ┘
```

**Linear layer:** `nn.Linear(in_features=2, out_features=3)` has weight matrix shape `(3, 2)` — 3 output neurons, each with 2 input weights.

**Batching:** Instead of one vector, feed a matrix of shape `(batch_size, in_features)` — each **row** is one example. One matrix multiply processes the entire batch in parallel.

> 🔬 Deep Dive
>
> Matrix multiplication is **not** element-wise. Entry \((i, j)\) of the product \(AB\) is the dot product of row \(i\) of \(A\) with column \(j\) of \(B\). That is why inner dimensions must match: \((m \times n)(n \times p) = (m \times p)\).

---

## 3. Formal Definitions

### Matrix

An \(m \times n\) **matrix** \(A\) has \(m\) rows and \(n\) columns:

\[
A = \begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}
\]

| Symbol | Meaning |
|--------|---------|
| \(a_{ij}\) | Entry in row \(i\), column \(j\) |
| \(m\) | Number of rows |
| \(n\) | Number of columns |
| \(A \in \mathbb{R}^{m \times n}\) | Real matrix with \(m\) rows, \(n\) columns |

### Matrix-vector multiplication

If \(W \in \mathbb{R}^{m \times n}\) and \(x \in \mathbb{R}^n\), then \(y = Wx \in \mathbb{R}^m\):

\[
y_i = \sum_{j=1}^{n} W_{ij} x_j
\]

Row \(i\) of \(W\) dotted with \(x\).

> **Plain English**
> Each output number is one row of the matrix dotted with the input vector.

> **Python**
> `y = W @ x`

### Matrix-matrix multiplication

If \(A \in \mathbb{R}^{m \times n}\) and \(B \in \mathbb{R}^{n \times p}\):

\[
C = AB \in \mathbb{R}^{m \times p}, \quad C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}
\]

**Shape rule:** Inner dimensions must match. \((m \times \mathbf{n})(\mathbf{n} \times p)\).

> **Plain English**
> Entry \((i,j)\) of the product is row \(i\) of the first matrix dotted with column \(j\) of the second.

> **Python**
> `C = A @ B`  *(check `A.shape[1] == B.shape[0]`)*

### Transpose

\(A^\top\) flips rows and columns. If \(A\) is \(m \times n\), then \(A^\top\) is \(n \times m\).

In attention: \(\mathbf{Q}\mathbf{K}^\top\) means multiply \(\mathbf{Q}\) (shape \((\text{seq}, d)\)) by \(\mathbf{K}^\top\) (shape \((d, \text{seq})\)) to get scores shape \((\text{seq}, \text{seq})\).

> **Plain English**
> Flip rows and columns — what was a row becomes a column.

> **Python**
> `A.T`

### Identity matrix

\(I\) has 1s on the diagonal, 0s elsewhere. \(Ix = x\) — multiplication does nothing. Useful for theoretical derivations.

---

## 4. Programming Perspective

NumPy uses `@` for matrix multiplication (Python 3.5+). `*` is element-wise — a frequent bug source.

| Mathematics | Python |
|-------------|--------|
| \(C = AB\) | `C = A @ B` |
| \(y = Wx\) | `y = W @ x` |
| \(A^\top\) | `A.T` |
| Shape \((m, n)\) | `A.shape == (m, n)` |

```python
import numpy as np

W = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
])  # shape (3, 2) — 3 outputs, 2 inputs

x = np.array([1, 0])  # shape (2,)

y = W @ x
print("W shape:", W.shape)
print("x shape:", x.shape)
print("y =", y)           # [1, 3, 5] — first column of W
print("y shape:", y.shape)  # (3,)
```

**Batch matrix multiply:** Each row of \(X\) is one example.

```python
X = np.array([
    [1, 0],
    [0, 1],
    [1, 1],
])  # shape (3, 2) — batch of 3

Y = X @ W.T  # (3, 2) @ (2, 3) = (3, 3) — wait, check dimensions
```

Actually for linear layer: \(Y = X W^\top + b\) when \(W\) is stored as `(out, in)` in PyTorch.

```python
# PyTorch convention: nn.Linear stores weight as (out_features, in_features)
W_pt = W  # (3, 2)
X_batch = X  # (3, 2)
Y = X_batch @ W_pt.T  # (3, 2) @ (2, 3) = (3, 3)
print(Y)
```

Each row of output is the linear layer applied to the corresponding input row.

---

## 5. Visualizations

Matrices are harder to draw than vectors, but we can visualize what a \(2 \times 2\) matrix does to the unit square.

```python
import numpy as np
import matplotlib.pyplot as plt

# Transformation matrix: scale x by 2, y by 0.5
T = np.array([[2, 0], [0, 0.5]])

# Unit square corners
square = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
transformed = T @ square

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(square[0], square[1], "b-o", label="original")
ax.plot(transformed[0], transformed[1], "r-o", label="after T")
ax.set_aspect("equal")
ax.legend()
ax.set_title("Matrix T transforms the unit square")
ax.grid(True, alpha=0.3)
plt.show()
```

**How to read this plot:** The blue square becomes a red rectangle — stretched horizontally (×2) and compressed vertically (×0.5). A linear layer learns a transformation matrix (plus bias) to map inputs to useful representations.

```python
# Heatmap of a small attention-like score matrix
np.random.seed(0)
Q = np.random.randn(4, 3)
K = np.random.randn(4, 3)
scores = Q @ K.T  # (4, 3) @ (3, 4) = (4, 4)

plt.figure(figsize=(5, 4))
plt.imshow(scores, cmap="coolwarm")
plt.colorbar(label="score")
plt.xlabel("key position")
plt.ylabel("query position")
plt.title("Q @ K.T — attention score matrix")
plt.show()
```

Each cell is a dot product between one query vector and one key vector — how much position \(i\) should attend to position \(j\).

---

## 6. Worked Examples

### Example 1: Matrix-vector multiply by hand

\[
W = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad x = \begin{bmatrix} 5 \\ 6 \end{bmatrix}
\]

**Row 0:** \(1 \times 5 + 2 \times 6 = 5 + 12 = 17\)

**Row 1:** \(3 \times 5 + 4 \times 6 = 15 + 24 = 39\)

**Result:** \(y = [17, 39]\)

```python
W = np.array([[1, 2], [3, 4]])
x = np.array([5, 6])
print(W @ x)  # [17 39]
```

### Example 2: Shape check

\(A\) is \(2 \times 3\), \(B\) is \(3 \times 2\). Can we compute \(AB\)?

**Step 1:** Inner dimensions: \(3 = 3\) — yes.

**Step 2:** Output shape: \((2 \times 2)\).

```python
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[7, 8], [9, 10], [11, 12]])
print((A @ B).shape)  # (2, 2)
print(A @ B)
```

### Example 3: Linear layer forward pass

One neuron layer: 2 inputs → 3 outputs.

```python
W = np.array([[0.5, -1.0],
              [0.1,  0.3],
              [2.0,  0.0]])  # (3, 2)
b = np.array([0.1, -0.2, 0.5])  # (3,)
x = np.array([1.0, 2.0])

y = W @ x + b
print(y)
```

### Example 4: Attention scores (simplified)

Sequence length 3, embedding dim 2.

```python
Q = np.array([[1, 0], [0, 1], [1, 1]])  # (3, 2)
K = np.array([[1, 0], [0, 1], [1, 0]])  # (3, 2)

scores = Q @ K.T  # (3, 3)
print(scores)
# scores[i,j] = dot product of query i with key j
```

Softmax is applied row-wise next (probability chapter) to get attention weights.

### Example 5: Batched multiply

```python
batch = np.array([
    [1, 0],
    [0, 1],
])  # (2, 2) — 2 examples

W = np.array([[1, 2], [3, 4]])  # (2, 2) — square for simplicity

# Each row of batch times W.T
out = batch @ W.T
print(out)
```

---

## 7. AI Connection

> 🧠 AI Insight
>
> A transformer block is mostly matrix multiplications, layer norms, and nonlinearities. FlashAttention and tensor cores exist because matmul dominates compute. Learning matrix shapes lets you read model code and papers without getting lost.

**`nn.Linear(in, out)`:**

- Weight: `(out, in)`
- Bias: `(out,)`
- Input: `(..., in)` — any leading dimensions are batch
- Output: `(..., out)`
- Computation: `output = input @ weight.T + bias`

**Embedding lookup + linear stack:**

```python
# Conceptual: token_ids -> embeddings -> linear layers
# embeddings: (batch, seq, d_model)
# W: (d_ff, d_model)
# output: (batch, seq, d_ff) via matmul on last dimension
```

**Attention (full formula — preview until transformers):**

> 📌 Preview — optional for now
>
> **Term:** scaled dot-product attention  
> **One line:** matrix multiply `Q @ K.T`, then softmax, then multiply by `V`  
> **Learn properly in:** [Attention Mechanism](../04-transformers/01-attention-mechanism.md)  
> Matmul shapes below are the core lesson; softmax details are in [Probability](06-probability.md).

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
\]

Three matrix multiplies: \(QK^\top\), then weights times \(V\).

**Weight initialization:** Matrices start random (Xavier, Kaiming) so dot products neither vanish nor explode at layer 1.

**Gradients:** Backprop through a linear layer involves multiplying by the transpose of the weight matrix — the same shapes you learn here, reversed.

---

## 8. Common Mistakes

> ⚠️ Common Mistake
>
> **Using `*` instead of `@` in NumPy.** `A * B` multiplies element-wise (requires broadcastable shapes). `A @ B` is matrix multiply. In PyTorch, `@` and `torch.matmul` are matrix multiply; `*` is element-wise.

> ⚠️ Common Mistake
>
> **Transposing when you should not (or vice versa).** PyTorch `nn.Linear` stores `weight` as `(out, in)`. You multiply `input @ weight.T`. Forgetting `.T` gives a shape error or silent bug if dimensions are square.

> ⚠️ Common Mistake
>
> **Ignoring batch dimensions.** Shape `(32, 128, 768)` means batch 32, sequence 128, features 768. Matmul operates on the **last** dimensions; batch dimensions broadcast. Print `.shape` constantly while debugging.

> ⚠️ Common Mistake
>
> **Assuming matrix multiplication is commutative.** \(AB \neq BA\) in general. Order matters — composition of transformations, order of layers, \(QK^\top\) vs \(KQ^\top\) are completely different.

**Correct understanding:** Check shapes before every multiply. Inner dimensions must align. Transpose swaps rows and columns. Linear layers and attention are dot products arranged as matrix multiply.

---

## 9. Exercises

### Easy

1. What is the shape of a matrix with 4 rows and 7 columns?
2. Can you multiply a \(3 \times 5\) matrix by a \(5 \times 2\) matrix? What is the output shape?
3. Compute \(\begin{bmatrix} 1 & 2 \end{bmatrix} \begin{bmatrix} 3 \\ 4 \end{bmatrix}\) by hand.
4. In NumPy, create a \(2 \times 3\) matrix and a \(3 \times 1\) vector. Multiply with `@` and print the result shape.

### Medium

5. Implement \(y = Wx + b\) for batch input \(X\) shape `(batch, in)` without a loop.
6. Given `A.shape == (2, 3)` and `B.shape == (2, 3)`, can you compute `A @ B.T`? What is the output shape?
7. Explain why `nn.Linear(768, 768)` has \(768 \times 768\) weights plus 768 biases.
8. For attention with `seq=8`, `d_k=64`, what are the shapes of `Q`, `K`, and `Q @ K.T`?

### Hard

9. Show that \((AB)^\top = B^\top A^\top\). Why does this matter for backprop?
10. Multiply a \(2 \times 3\) matrix by a \(3 \times 2\) matrix. Compare `A @ B` vs `B @ A` — are they the same shape? Same values?

### Challenge (optional — includes previews)

11. *(Optional)* Plot the effect of rotation matrix \(\begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}\) on a unit square for \(\theta = \pi/4\). Trig refresher: [Math Basics](../00-intro/05-math-basics.md)

12. **Matrix playground:** Write a function that takes two compatible matrices, prints the shape rule step-by-step, computes the product, and highlights one entry \(C_{ij}\) showing which row and column were dotted.

13. **Mini attention:** With random `Q, K, V` of shapes `(4, 8)`, compute `softmax(Q @ K.T / sqrt(8), axis=-1) @ V` using NumPy. Print output shape and verify rows of the softmax matrix sum to 1.

---

## 10. Mini Project

### Matrix Multiplication Explorer

Build a tool that:

1. Accepts two matrices \(A\) and \(B\) with compatible shapes.
2. Prints shapes and the resulting shape rule.
3. Computes \(AB\) and displays as a heatmap.
4. Demonstrates one linear layer: random `W`, `b`, batch `X`, output `X @ W.T + b`.
5. Saves heatmap to `book/assets/05-matrices-explorer.png`.

```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

A = np.array([[1, 2, 3], [4, 5, 6]])      # (2, 3)
B = np.array([[7, 8], [9, 10], [11, 12]])  # (3, 2)
C = A @ B

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(C, cmap="Blues")
for i in range(C.shape[0]):
    for j in range(C.shape[1]):
        ax.text(j, i, f"{C[i,j]:.0f}", ha="center", va="center")
plt.colorbar(im)
plt.title(f"A{C.shape} = A{A.shape} @ B{B.shape}")
out = Path("book/assets/05-matrices-explorer.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
```

<details>
<summary>Mini project checklist</summary>

- [ ] Shape rule printed before multiply
- [ ] Product computed and visualized
- [ ] Linear layer batch example included
- [ ] Figure saved to `book/assets/`

</details>

---

## 11. Interview Questions

**Q1:** What does matrix multiplication represent in a neural network layer?

**A1:** Each output neuron computes a weighted sum of inputs — a dot product between the input vector and a row of the weight matrix. Stacking all neurons, the layer computes `y = Wx + b` (with appropriate transposes per framework convention). The matrix encodes all weights simultaneously; batching applies the same transformation to many input vectors in one operation.

**Q2:** Why must inner dimensions match for matrix multiplication?

**A2:** Each entry \(C_{ij}\) is the dot product of row \(i\) of \(A\) with column \(j\) of \(B\). Dot products require equal-length vectors. Row length of \(A\) is its number of columns \(n\); column length of \(B\) is its number of rows \(n\). Those must match. The outer dimensions \((m, p)\) become the result shape.

**Q3:** What is the shape of the attention score matrix in self-attention?

**A3:** For sequence length \(L\) and key/query dimension \(d_k\), \(Q\) and \(K\) have shape \((L, d_k)\). The product \(QK^\top\) has shape \((L, L)\). Entry \((i, j)\) is the score between query position \(i\) and key position \(j\) — how much position \(i\) should attend to position \(j\) before softmax.

**Q4:** Is matrix multiplication commutative?

**A4:** No. \(AB \neq BA\) in general — different shapes may even make one product undefined. In networks, layer order matters: \(W_2(W_1 x) \neq W_1(W_2 x)\) in general. Composition of transformations corresponds to multiplying matrices in a specific order.

**Q5:** How does GPU batching relate to matrix multiplication?

**A5:** GPUs excel at large, regular matrix multiplies. Batching stacks independent examples into extra rows or batch dimensions so one kernel launch processes the entire mini-batch. Frameworks broadcast over leading dimensions so `(..., m, n) @ (n, p)` works for arbitrary batch prefixes — critical for training throughput.

---

## 12. Summary

### Key formulas

| Concept | Formula / Rule |
|---------|----------------|
| Matrix-vector | \(y_i = \sum_j W_{ij} x_j\) |
| Matrix-matrix | \(C_{ij} = \sum_k A_{ik} B_{kj}\) |
| Shape rule | \((m \times \mathbf{n})(\mathbf{n} \times p) = (m \times p)\) |
| Linear layer | \(y = Wx + b\) (check transpose convention) |
| Attention scores | \(\text{scores} = QK^\top\) |
| Transpose product | \((AB)^\top = B^\top A^\top\) |

### Key terminology

- **Matrix** — rectangular grid of numbers
- **Row / column** — horizontal / vertical entries
- **Shape** — `(rows, columns)`
- **Matrix multiplication** — dot products of rows with columns
- **Transpose** — flip rows and columns (\(A^\top\))
- **Linear transformation** — function \(x \mapsto Wx\) (plus bias)
- **Batch dimension** — leading axes indexing independent examples

### Readiness checks

Before the next chapter, you should be able to:

1. State the shape rule \((m \times n)(n \times p) = (m \times p)\) and apply it to two example matrices.
2. Compute a small matrix-vector product by hand.
3. Write `y = W @ x + b` in NumPy for a batch of inputs without a Python loop.
4. Explain why `A @ B` is not the same as `B @ A` in general.
5. Predict the shape of `Q @ K.T` given sequence length and embedding dimension.

If any item is shaky, reread §3 and the [cheatsheet](05-matrices-cheatsheet.md).

---

## 13. Preview

Matrices transform vectors deterministically. Real models also need **uncertainty** and **probabilities** — predicting class distributions, sampling tokens, interpreting outputs as likelihoods. The next chapter — **Probability** — covers distributions, softmax (turning scores into probabilities), and cross-entropy loss.

Together, matrices and probability power the output layer of classifiers and language models.

**Next chapter:** [Probability](06-probability.md)

---

## Lab

Companion notebook: [`app/math/05_matrices.ipynb`](../../app/math/05_matrices.ipynb)

## Review

- Cheatsheet: [Matrices — Cheatsheet](05-matrices-cheatsheet.md)
- Jargon: [Vocabulary Roadmap](../00-intro/04-vocabulary-roadmap.md)
