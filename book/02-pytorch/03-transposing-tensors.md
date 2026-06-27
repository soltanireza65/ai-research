# Transposing Tensors

## 1. Introduction

Matrix multiplication has strict shape rules. Often the numbers you have are oriented the wrong way — rows where columns should be, batch axes in the wrong order. **Transposing** swaps dimensions: rows become columns, leading axes move to the back. In PyTorch, `.T` or `.transpose()` does this in one line.

You cannot read transformer code without transpose: `scores = Q @ K.transpose(-2, -1)`. You cannot debug linear layers without knowing weights are stored as `(out, in)` while inputs arrive as `(batch, in)`. Transpose is the shape-fixing tool that makes matmul work.

After this chapter you will be able to:

- Transpose 2D matrices with `.T` and general tensors with `.transpose()` / `.permute()`.
- Predict the shape after any transpose operation.
- Use transpose correctly in attention and linear layer code.
- Distinguish `.T` from `.reshape()` and `.view()`.

**Where this appears in AI:** Attention computes `Q @ K^T`. Weight matrices are transposed in `F.linear`. Loss functions compare predictions and targets with different orientations. Convolution and batch normalization use permute to switch between `NCHW` and `NHWC` layouts.

Transpose is not "advanced math" — it is bookkeeping. Models store parameters in one layout; equations assume another. Transpose is the adapter between them, like converting JSON keys from camelCase to snake_case before two APIs can talk.

When reading a paper that writes \(W^\top x\), the authors assume column vectors. PyTorch uses row batches `(batch, features)` and stores weights as `(out, in)`, so the implemented multiply is `x @ W.T`. Same mathematics, different storage convention — transpose bridges the gap.

---

## 2. Intuition

> 💡 Intuition
>
> Transpose is **flipping a spreadsheet along its main diagonal**. What was in row 3, column 7 moves to row 7, column 3. For a 2D matrix, `.T` swaps the two axes. For higher-dimensional tensors, you choose **which pair** of dimensions to swap.

```
  Original A (2×3):          Transpose A.T (3×2):

  row 0: [1  2  3]           col 0: [1  4]
  row 1: [4  5  6]           col 1: [2  5]
                             col 2: [3  6]
```

Element at position \((i, j)\) in \(A\) moves to \((j, i)\) in \(A^\top\).

In attention, each **row** of \(Q\) is one token's query vector. Each **row** of \(K\) is one token's key vector. To compare query \(i\) with key \(j\), we need dot products between rows of \(Q\) and rows of \(K\). That is `Q @ K.T`: rows of \(Q\) dot rows of \(K^\top\) (which are columns of \(K\)).

> 🔬 Deep Dive
>
> For 1D tensors, `.T` does nothing — there is only one axis. For 3D+ tensors, `.T` reverses **all** dimensions (NumPy convention). Prefer explicit `.transpose(dim0, dim1)` or `.permute(...)` for clarity in production code.

---

## 3. Formal Definitions

### Transpose of a 2D matrix

If \(A\) has shape \((m, n)\), then \(A^\top\) (or `A.T`) has shape \((n, m)\):

\[
(A^\top)_{ij} = A_{ji}
\]

### General transpose

For a tensor with shape \((d_0, d_1, \ldots, d_{k-1})\), swapping dimensions \(a\) and \(b\):

\[
\text{shape after .transpose}(a, b) = (\ldots, d_b \text{ at position } a, \ldots, d_a \text{ at position } b, \ldots)
\]

### permute

`.permute(i, j, k, ...)` reorders **all** dimensions at once. If `x.shape == (2, 3, 4)`:

```python
x.permute(2, 0, 1)  # shape (4, 2, 3)
```

---

## 4. Programming Perspective

| Goal | PyTorch |
|------|---------|
| 2D transpose | `A.T` or `A.transpose(0, 1)` |
| Swap two dims | `x.transpose(0, 2)` |
| Full reorder | `x.permute(2, 0, 1)` |
| Last two dims only | `x.transpose(-2, -1)` |

```python
import torch

A = torch.arange(6).reshape(2, 3).float()
print("A:\n", A)
print("A.T:\n", A.T)
print("A shape:", A.shape)    # torch.Size([2, 3])
print("A.T shape:", A.T.shape)  # torch.Size([3, 2])
```

```python
# 3D example: batch of matrices
batch = torch.randn(4, 2, 3)   # 4 matrices, each 2×3
swapped = batch.transpose(-2, -1)  # (4, 3, 2) — swap last two dims
print(batch.shape, "->", swapped.shape)
```

### Attention-style transpose

```python
Q = torch.randn(2, 8, 64)  # batch=2, seq=8, dim=64
K = torch.randn(2, 8, 64)
scores = Q @ K.transpose(-2, -1)  # (2, 8, 8)
print(scores.shape)
```

`K.transpose(-2, -1)` turns `(2, 8, 64)` into `(2, 64, 8)` — wait, that's wrong!

Actually: `K` is `(2, 8, 64)`. `K.transpose(-2, -1)` swaps dims -2 and -1: `(2, 64, 8)`. Then `Q @ K.T` would be `(2, 8, 64) @ (2, 64, 8)` — batch matmul gives `(2, 8, 8)`. Correct.

---

## 5. Visualizations

```python
import torch
import matplotlib.pyplot as plt

A = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

for ax, M, title in zip(axes, [A, A.T], ["A (2×3)", "A.T (3×2)"]):
    im = ax.imshow(M.numpy(), cmap="coolwarm", aspect="auto")
    ax.set_title(title)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{int(M[i,j].item())}", ha="center", va="center")
    ax.set_xlabel("column index")
    ax.set_ylabel("row index")

plt.tight_layout()
plt.show()
```

**How to read:** Left panel rows become right panel columns. Row 0 `[1,2,3]` of \(A\) becomes column 0 of \(A.T\). The diagonal (1, 5) stays fixed — transpose reflects across the diagonal.

```python
# Visualize permute on a small 3D tensor
x = torch.arange(24).reshape(2, 3, 4)
y = x.permute(2, 1, 0)  # (4, 3, 2)
print("original:", x.shape, "permute(2,1,0):", y.shape)
# Same 24 elements, different organization
```

---

## 6. Worked Examples

### Example 1: Basic 2D transpose

```python
A = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])  # (3, 2)
print(A.T)  # (2, 3)
# tensor([[1., 3., 5.],
#         [2., 4., 6.]])
```

### Example 2: Making matmul work

We have `x` shape `(4, 10)` and want `W` shape `(10, 5)` so `x @ W` gives `(4, 5)`. If weights were stored as `(5, 10)`:

```python
x = torch.randn(4, 10)
W_stored = torch.randn(5, 10)  # nn.Linear layout
y = x @ W_stored.T  # (4, 10) @ (10, 5) -> (4, 5)
print(y.shape)
```

### Example 3: 3D transpose along dims 0 and 2

```python
t = torch.arange(24).reshape(2, 3, 4)
t_perm = t.transpose(0, 2)  # swap dim 0 and dim 2
print(t.shape, "->", t_perm.shape)  # (2,3,4) -> (4,3,2)
```

### Example 4: Symmetric matrix

If \(A = A^\top\), the matrix is **symmetric**. Covariance matrices and some attention variants use symmetry.

```python
A = torch.tensor([[1., 2.], [2., 3.]])
print(torch.allclose(A, A.T))  # True
```

### Example 5: Step-by-step attention transpose

Consider `Q` and `K` with shape `(batch=2, seq=4, d_k=3)`:

```python
import torch

Q = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
K = Q.clone()  # simplified: identical for illustration
Kt = K.transpose(-2, -1)
print("Q:", Q.shape, "K^T:", Kt.shape)
scores = Q @ Kt
print("scores:", scores.shape)  # (2, 4, 4)
```

**Step 1:** `K` is `(2, 4, 3)` — batch 2, 4 tokens, 3 features per token.

**Step 2:** `transpose(-2, -1)` swaps seq and feature → `(2, 3, 4)`.

**Step 3:** Batch matmul `(2, 4, 3) @ (2, 3, 4)` → `(2, 4, 4)`. Entry `(b, i, j)` is the dot product between query token `i` and key token `j` in batch `b`.

### Example 6: permute for NCHW → NHWC

```python
img = torch.randn(8, 3, 64, 64)
hwc = img.permute(0, 2, 3, 1)
print(img.shape, hwc.shape)  # (8,3,64,64) -> (8,64,64,3)
```

Some visualization and TensorFlow-interop pipelines expect channels last; `permute` is the explicit conversion.

### Example 7: Verify (AB)^T = B^T A^T

```python
A = torch.randn(3, 4)
B = torch.randn(4, 5)
lhs = (A @ B).T
rhs = B.T @ A.T
print(torch.allclose(lhs, rhs, atol=1e-5))
```

This identity is used when deriving backprop rules for weight gradients in linear layers.

Transposing is inexpensive when implemented as a view, but the next operation may force a contiguous copy — profile before optimizing.

---

## 7. AI Connection

> 🧠 AI Insight
>
> In scaled dot-product attention, `Q @ K.transpose(-2, -1)` produces a `(seq, seq)` matrix of compatibility scores. Without transpose, inner dimensions would not align for matmul. The `/ sqrt(d_k)` scaling comes after this multiply.

**Linear layers:** Weights `(out, in)`; inputs `(batch, in)`; use `W.T` in the multiply.

**Embeddings + projection:** Token embeddings `(batch, seq, dim)` may be transposed or permuted for certain efficient kernels (e.g. `(batch, dim, seq)` for conv1d-style processing).

**Loss computation:** Predictions `(batch, num_classes)` vs one-hot targets `(batch, num_classes)` — sometimes one is transposed for broadcasting in custom losses.

**Image tensors:** PyTorch conv uses `(N, C, H, W)`. Some visualization code wants `(N, H, W, C)` — `permute(0, 2, 3, 1)`.

**Backprop:** Transpose is its own inverse for gradient flow: if forward uses `A.T`, backward transposes again. Autograd handles this automatically.

**Efficient kernels:** Some optimized attention implementations fuse transpose with matmul internally. When reading profiler traces, a standalone `transpose` before `bmm` may disappear in fused builds — the math is unchanged.

**Export and ONNX:** Static export tools record transpose as explicit ops. Changing `.transpose(-2,-1)` to `.T` on 3D tensors can break exported graphs if the traced model assumed different dimension reversal.

---

## 8. Common Mistakes

> ⚠️ Common Mistake
>
> **Using `.T` on 1D tensors expecting a row/column vector.** A 1D tensor `(n,)` has no second axis to swap. Use `.unsqueeze(0)` or `.unsqueeze(1)` to create a row or column vector first.

> ⚠️ Common Mistake
>
> **Confusing `.T` on 3D+ with swapping only the last two dims.** For 3D, `.T` reverses all three dimensions: `(a,b,c) → (c,b,a)`. For attention you usually want `.transpose(-2, -1)`, not `.T`.

> ⚠️ Common Mistake
>
> **Forgetting that transpose can make tensors non-contiguous.** Some ops require contiguous memory. After transpose, call `.contiguous()` before `.view()` if you get errors.

> ⚠️ Common Mistake
>
> **Double transpose in attention.** `K.transpose(-2, -1)` once is correct. `K.T` on 3D may do something different than you expect.

**Correct understanding:** Use explicit dim arguments for 3D+. Check `.shape` after every transpose. Use `.contiguous()` when chaining with reshape.

When debugging transpose bugs, print shapes **before and after** each operation in the forward pass. A one-line shape log often reveals whether you swapped sequence and feature axes or reversed the entire batch dimension by mistake.

---

## 9. Exercises

### Easy

1. Create a `(3, 5)` tensor. Print `.T` and its shape.
2. Verify `(A.T).T == A` for a random `(4, 7)` matrix.
3. Create a row vector `(1, 6)` and column vector `(6, 1)` from the same 1D tensor using `unsqueeze` and transpose.

### Medium

4. Given `Q`, `K` both `(2, 10, 32)`, compute attention scores of shape `(2, 10, 10)`.
5. `x` is `(batch, features)`. `W` is `(out, features)`. Write the matmul using `@` and `.T`.
6. Permute `(2, 3, 4, 5)` to `(5, 4, 3, 2)` using `.permute()`.
7. After `x.transpose(1, 2)`, explain why `x.view(-1)` might fail and how `.contiguous()` helps.

### Hard

8. Show that `(AB)^T = B^T A^T` with random `(3, 4)` and `(4, 5)` matrices in PyTorch.
9. Implement batched `K.transpose(-2, -1)` for `K` shape `(8, 12, 64)` and verify output shape `(8, 64, 12)`.
10. When would a weight matrix be symmetric in a neural network? Give one example or explain why it is rare.

### Challenge

11. **Attention score heatmap:** Random `Q`, `K` with 8 tokens, 16 dims. Compute scores, plot as 8×8 heatmap. Label axes "query token" and "key token".
12. **Layout converter:** Function `nchw_to_nhwc(t)` and inverse using only `permute`.

---

## 10. Mini Project

### Transpose Explorer

Build a utility that takes a tensor, applies a list of transpose/permute operations, and logs shape at each step.

```python
import torch

def explore_transpose(x, ops):
    """ops: list of ('transpose', d0, d1) or ('permute', *dims)"""
    print(f"start: {tuple(x.shape)}")
    for op in ops:
        if op[0] == "transpose":
            x = x.transpose(op[1], op[2])
        elif op[0] == "permute":
            x = x.permute(*op[1:])
        print(f"  after {op}: {tuple(x.shape)}")
    return x

x = torch.randn(2, 8, 64)
explore_transpose(x, [
    ("transpose", -2, -1),
])
# Simulates K.T for attention
scores = torch.randn(2, 8, 64) @ x  # placeholder Q @ K.T
print("scores shape:", scores.shape)
```

<details>
<summary>Mini project checklist</summary>

- [ ] 2D and 3D examples
- [ ] Shape logged after each op
- [ ] Attention-style example included

</details>

---

## 11. Interview Questions

**Q1:** What does `A.T` do for a 2D tensor?

**A1:** It swaps rows and columns. If `A` has shape `(m, n)`, `A.T` has shape `(n, m)`. Element at `(i, j)` in `A` moves to `(j, i)` in `A.T`. It is the matrix transpose from linear algebra.

**Q2:** Why do we write `K.transpose(-2, -1)` in attention instead of `K.T`?

**A2:** For 3D tensors `(batch, seq, dim)`, `.T` reverses all three dimensions to `(dim, seq, batch)`, which is usually wrong. We only want to swap the last two axes: `K.transpose(-2, -1)` maps `(batch, seq, dim)` to `(batch, dim, seq)`. Then `Q @ K.transpose(-2, -1)` yields `(batch, seq, seq)` attention scores — each query position against each key position.

**Q3:** What is the difference between `transpose` and `permute`?

**A3:** `transpose(a, b)` swaps two dimensions. `permute(d0, d1, ...)` specifies the full new order of all dimensions. Use `permute` when reordering more than two axes (e.g. `NCHW` to `NHWC`).

**Q4:** What does non-contiguous mean after transpose?

**A4:** PyTorch tensors store data in a flat buffer with **strides** defining how to step through dimensions. Transpose often only swaps stride metadata without copying data, so memory layout no longer matches the logical row-major order. The tensor is **non-contiguous**. Operations like `.view()` require contiguous layout; use `.contiguous().view(...)` or `.reshape()` which may copy if needed.

**Q5:** How does transpose affect gradients in backprop?

**A5:** Transpose is a linear rearrangement. In backward pass, gradients flow through the inverse transpose — transpose again. If forward computes `Y = X.T`, then `dL/dX = (dL/dY).T`. Autograd implements this automatically when you call `.backward()`.

### Extended pattern: multi-head layout

In production transformer code, tensors often pass through this sequence:

1. Start with `(batch, seq, d_model)`.
2. `reshape` to `(batch, seq, num_heads, d_k)`.
3. `transpose` to `(batch, num_heads, seq, d_k)` for batched attention matmuls.
4. After attention, `transpose` back and `reshape` to `(batch, seq, d_model)`.

Each transpose is purposeful — it places `num_heads` next to batch for parallel GPU kernels. Reading open-source LLM implementations requires recognizing this idiom immediately.

---

## 12. Summary

### Key formulas

| Operation | Shape change |
|-----------|--------------|
| 2D `.T` | `(m, n) → (n, m)` |
| `.transpose(a, b)` | swap dims `a` and `b` |
| `.permute(p0, p1, ...)` | full reorder |
| `(AB)^T = B^T A^T` | order reverses |

### Key terminology

- **Transpose** — swap rows and columns (2D) or swap dimension pairs
- **permute** — arbitrary dimension reordering
- **Contiguous** — memory layout matches logical order
- ** strides** — step sizes for each dimension in memory

---

## 13. Preview

Transpose swaps axes but keeps the same total number of elements. Often you need to **change shape** without changing data — flatten a batch, add a channel dimension, split heads. The next chapter covers **reshaping**: `reshape`, `view`, `flatten`, `unsqueeze`, and `squeeze`.

**Next chapter:** [Reshaping Tensors](04-reshaping-tensors.md)

---

## Lab

Companion notebook: [`app/pytorch/03_transposing_tensors.ipynb`](../../app/pytorch/03_transposing_tensors.ipynb)
