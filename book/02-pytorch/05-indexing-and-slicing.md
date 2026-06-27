# Indexing and Slicing

## 1. Introduction

Tensors hold millions of numbers; you rarely need all of them at once. **Indexing** picks one element. **Slicing** picks rows, columns, ranges, or sub-blocks. **Boolean masking** selects every position where a condition is true. These operations are how you implement padding masks in transformers, extract the last valid token, and debug a single misclassified example.

PyTorch indexing mirrors NumPy: `x[0]`, `x[:, 2]`, `x[1:4]`, `x[x > 0]`. The rules extend naturally to任意 dimensions. Master slicing and you can read and write any tensor shape without reshaping everything.

After this chapter you will be able to:

- Index and slice tensors along any dimension.
- Select rows, columns, and submatrices with `:` notation.
- Apply boolean and integer (fancy) indexing.
- Build attention masks and extract batch elements confidently.

**Where this appears in AI:** Padding masks zero out invalid tokens. You slice `hidden[:, -1, :]` for sentence classification. You index `logits[range(batch), labels]` for cross-entropy. Data loading indexes samples by batch indices.

If NumPy slicing felt mysterious, PyTorch uses the same rules — including half-open intervals where `stop` is excluded. The muscle memory you build here transfers directly to NumPy preprocessing pipelines.

**Advanced indexing and autograd:** Some fancy indexing patterns have limited or no gradient support for certain index tensors. Prefer `gather`, `index_select`, or batched ops when writing custom autograd functions intended for training.

Indexing is also how you **inspect** models during development: pull one batch element, one head, one channel — narrow failures before they disappear into batch averages. Treat slicing as your primary debugger for tensor-shaped data.

---

## 2. Intuition

> 💡 Intuition
>
> Indexing is **coordinates in a spreadsheet**. For a 2D tensor, `[row, col]` is one cell. `:` means "all" along that axis — `[1, :]` is entire row 1; `[:, 2]` is entire column 2. Slicing `[start:stop]` takes a range (stop excluded, Python-style).

```
  Tensor (4×5):

       col0 col1 col2 col3 col4
  row0 [ 0    1    2    3    4 ]
  row1 [ 5    6    7    8    9 ]
  row2 [10   11   12   13   14 ]
  row3 [15   16   17   18   19 ]

  x[1, :]     → row 1:     [5, 6, 7, 8, 9]
  x[:, 2]     → column 2:  [2, 7, 12, 17]
  x[1:3, 2:4] → top-left 2×2 of sub-block starting at (1,2):
                [[7, 8], [12, 13]]
```

For 3D tensors `(batch, seq, feature)`, `x[0]` is the first batch item; `x[:, 0, :]` is the first token of every sequence in the batch; `x[0, -1, :]` is the last token of the first sequence.

> 🔬 Deep Dive
>
> Negative indices count from the end: `-1` is the last row/token/feature. `x[::-1]` reverses along the first dimension. Slicing returns a **view** when strides allow — mutations may affect the parent tensor.

---

## 3. Formal Definitions

### Integer indexing

For tensor \(T\) with shape \((d_0, d_1, \ldots, d_{k-1})\), index tuple \((i_0, i_1, \ldots, i_{k-1})\) selects one element if each \(i_j\) is an integer.

### Slice notation

Slice `start:stop:step` along dimension \(j\) selects indices from `start` inclusive to `stop` exclusive, stepping by `step`.

### Boolean mask

Mask \(M\) with the same shape as \(T\) (or broadcastable). `T[M]` returns a 1D tensor of all \(T_{ij\ldots}\) where \(M_{ij\ldots} = \text{True}\).

---

## 4. Programming Perspective

```python
import torch

x = torch.arange(20).reshape(4, 5)
print(x)
print("row 1:", x[1])
print("col 2:", x[:, 2])
print("element (2,3):", x[2, 3])
print("submatrix:", x[1:3, 2:4])
```

```python
# Boolean mask
print("values > 10:", x[x > 10])

# Last row
print("last row:", x[-1])

# 3D example
batch = torch.randn(3, 4, 8)  # 3 sequences, 4 tokens, 8 dims
first_seq = batch[0]           # (4, 8)
all_first_tokens = batch[:, 0, :]  # (3, 8)
last_token_first_seq = batch[0, -1, :]  # (8,)
```

```python
# Fancy indexing with integer tensor
indices = torch.tensor([0, 2, 3])
selected_rows = x[indices]  # rows 0, 2, 3
print(selected_rows.shape)  # (3, 5)
```

### Attention mask pattern

```python
seq_len = 5
# True = keep, False = mask (varies by convention)
mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
scores = torch.randn(seq_len, seq_len)
scores_masked = scores.masked_fill(~mask, float("-inf"))
print(scores_masked)
```

---

## 5. Visualizations

```python
import torch
import matplotlib.pyplot as plt

x = torch.arange(20).reshape(4, 5)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Full matrix
axes[0].imshow(x.numpy(), cmap="Blues")
axes[0].set_title("Full 4×5")
for i in range(4):
    for j in range(5):
        axes[0].text(j, i, str(x[i,j].item()), ha="center", va="center")

# Highlight row 1
mask_row = torch.zeros(4, 5)
mask_row[1, :] = 1
axes[1].imshow(mask_row.numpy(), cmap="Reds", alpha=0.5)
axes[1].imshow(x.numpy(), cmap="Blues", alpha=0.5)
axes[1].set_title("Row 1 selected")

# Highlight submatrix
mask_sub = torch.zeros(4, 5)
mask_sub[1:3, 2:4] = 1
axes[2].imshow(mask_sub.numpy(), cmap="Reds", alpha=0.5)
axes[2].imshow(x.numpy(), cmap="Blues", alpha=0.5)
axes[2].set_title("Submatrix [1:3, 2:4]")

for ax in axes:
    ax.set_xlabel("column")
    ax.set_ylabel("row")

plt.tight_layout()
plt.show()
```

**How to read:** Red overlay shows selected region. Row slice is one horizontal band; submatrix slice is a contiguous block — the same pattern used to crop image patches or extract token windows.

```python
# Histogram of masked values
values = torch.randn(1000)
positive = values[values > 0]
plt.figure(figsize=(6, 4))
plt.hist(positive.numpy(), bins=30, color="seagreen")
plt.xlabel("value")
plt.ylabel("count")
plt.title(f"Positive subset ({len(positive)} of 1000)")
plt.show()
```

---

## 6. Worked Examples

### Example 1: Top-left 2×2 submatrix

```python
x = torch.arange(20).reshape(4, 5)
sub = x[0:2, 0:2]
print(sub)
# tensor([[0, 1],
#         [5, 6]])
```

### Example 2: Select batch element and token

```python
hidden = torch.randn(32, 128, 768)  # batch, seq, hidden
sample_5 = hidden[5]                 # all tokens for example 5
last_tok = hidden[5, -1, :]          # last token vector, shape (768,)
first_tok_batch = hidden[:, 0, :]    # first token for all, (32, 768)
```

### Example 3: Boolean filter

```python
x = torch.tensor([1., -2., 3., -4., 5.])
positives = x[x > 0]
print(positives)  # tensor([1., 3., 5.])
```

### Example 4: Cross-entropy indexing

For logits `(N, C)` and label indices `(N,)`, pick the predicted score for the true class:

```python
logits = torch.randn(4, 10)
labels = torch.tensor([2, 5, 1, 9])
selected = logits[torch.arange(4), labels]
print(selected.shape)  # (4,)
```

### Example 5: Padding mask construction

```python
import torch

lengths = torch.tensor([5, 3, 7, 4])
max_len = 8
positions = torch.arange(max_len)
pad_mask = positions.unsqueeze(0) >= lengths.unsqueeze(1)
print(pad_mask)
# True where padded
```

**Step 1:** `positions` is `[0,1,...,7]`. **Step 2:** Compare each row's true length to positions. **Step 3:** Use mask in attention or loss to ignore pad tokens.

### Example 6: gather for per-row selection

```python
x = torch.randn(3, 5)
idx = torch.tensor([0, 4, 2])
out = x.gather(1, idx.unsqueeze(1)).squeeze(1)
manual = x[torch.arange(3), idx]
print(torch.allclose(out, manual))
```

`gather` is the exported-graph-friendly version of advanced indexing.

### Example 7: Narrow for sliding windows

```python
seq = torch.randn(1, 100, 512)
window = seq[:, 10:20, :]  # tokens 10..19
print(window.shape)  # (1, 10, 512)
```

Sliding-window attention uses slicing along sequence dimension instead of full quadratic attention.

### Example 8: index_copy for cache updates

```python
cache = torch.zeros(4, 10)
new_vals = torch.tensor([1., 2., 3., 4.])
positions = torch.tensor([2, 5, 7, 9])
cache.index_copy_(1, positions, new_vals.unsqueeze(0))
print(cache)
```

Scatter-style updates appear in KV cache implementations during incremental decoding.

### Example 9: Variable-length last token (not pad)

```python
hidden = torch.randn(3, 8, 64)
lengths = torch.tensor([5, 8, 3])
batch_idx = torch.arange(3)
last_idx = lengths - 1
last_hidden = hidden[batch_idx, last_idx, :]
print(last_hidden.shape)  # (3, 64)
```

When sequences are padded, `[:, -1, :]` returns pad positions for short sequences — index by `lengths - 1` instead.

Indexing is the primary interface between high-level model code and raw tensor memory. Every `forward` that selects a token, masks a pad position, or gathers a class logit relies on the patterns in this chapter.

---

## 7. AI Connection

> 🧠 AI Insight
>
> Causal (decoder-only) transformers use a **causal mask** so position \(i\) cannot attend to positions \(j > i\). Implemented as a boolean upper-triangular mask applied to attention scores before softmax — built with slicing and `torch.triu` or `tril`.

**Padding masks:** Sequences differ in length; pad tokens are masked with `attention_mask` so the model ignores them. Often `mask = (input_ids != pad_token_id)`.

**Classification:** BERT uses `[CLS]` token at index 0: `output[:, 0, :]`. GPT-style models often use last token: `output[:, -1, :]`.

**Gather for loss:** `F.cross_entropy` does this internally, but custom losses use advanced indexing.

**DataLoader:** Returns indexed batches — understanding slicing helps debug when batch order shuffles.

**Gradient indexing:** Slicing in forward creates corresponding gradient routes in backward — only selected elements receive gradients.

---

## 8. Common Mistakes

> ⚠️ Common Mistake
>
> **Confusing `x[1:3]` with `x[[1, 2]]`.** Slice `1:3` gives indices 1 and 2 (stop exclusive). List `[1, 2]` is fancy indexing — same here but different for non-contiguous picks and gradient behavior.

> ⚠️ Common Mistake
>
> **Modifying a slice unexpectedly affects the original.** Slices are often views. `x[0].fill_(0)` may zero the first row of `x`. Use `.clone()` if you need an independent copy.

> ⚠️ Common Mistake
>
> **Wrong dimension in 3D.** `x[:, 2]` vs `x[2, :]` — first fixes dim 0 (batch), second indexes dim 0 to 2. Always say dim names aloud: batch, seq, feature.

> ⚠️ Common Mistake
>
> **Boolean mask shape mismatch.** `x[mask]` requires mask to be broadcastable to `x`'s shape or match exactly.

**Correct understanding:** Name dimensions. Use `:` for full axes. Clone when mutating slices independently.

For transformer debugging, a useful checklist is: batch dimension first, sequence second, features last — `(B, T, D)`. Any slice that fixes batch but ranges over sequence (e.g. `[:, start:end, :]`) is probably implementing a window or mask; any slice that fixes sequence index selects token-level representations.

**Performance note:** Boolean masking followed by `nonzero()` materializes indices and can be slower than `torch.where` or fused kernels on GPU. For large tensors, prefer built-in masked operations (`masked_fill`, `masked_select`) over Python-level filtering loops.

---

## 9. Exercises

### Easy

1. Create `(4, 5)` arange tensor. Extract row 2, column 3, and element `(1, 4)`.
2. Select all values greater than 10 from a random tensor of shape `(10,)`.
3. Use negative index to get the last column of a matrix.

### Medium

4. Extract top-left 2×2 submatrix from `(4, 5)` tensor (lab exercise).
5. From `(8, 50, 256)` hidden states, get shape `(8, 256)` using the last token of each sequence.
6. Build a lower-triangular boolean mask for seq_len=6.
7. Use `torch.arange` and fancy indexing to select rows 0, 2, 4 from a matrix.

### Hard

8. Implement `gather_logits(logits, labels)` for `(N, C)` and `(N,)` without a loop.
9. Mask all padding positions in `scores (batch, heads, seq, seq)` given `pad_mask (batch, seq)`.
10. Explain view vs copy when slicing `x[0:2]` and mutating the result.

### Challenge

11. **Attention mask visualizer:** Plot 16×16 causal mask as heatmap. Label axes query/key positions.
12. **Batch debugger:** Given `(B, T, D)` and list of actual lengths per sequence, extract each sequence's last *valid* token (not pad).

---

## 10. Mini Project

### Sequence Slicer

Implement utilities for transformer hidden states:

```python
import torch

def first_token(hidden):
    """hidden: (B, T, D) -> (B, D)"""
    return hidden[:, 0, :]

def last_token(hidden):
    return hidden[:, -1, :]

def nth_token(hidden, n):
    return hidden[:, n, :]

def causal_mask(seq_len, device="cpu"):
    return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))

B, T, D = 4, 10, 64
h = torch.randn(B, T, D)
print("first:", first_token(h).shape)
print("last:", last_token(h).shape)
print("mask:", causal_mask(T).shape)
```

<details>
<summary>Mini project checklist</summary>

- [ ] First/last/nth token extractors
- [ ] Causal mask builder
- [ ] Docstrings with shape annotations

</details>

---

## 11. Interview Questions

**Q1:** What is the difference between `x[1:3]` and `x[[1, 2]]`?

**A1:** `x[1:3]` is slicing along the first dimension — contiguous range, typically a view. `x[[1, 2]]` is fancy (integer array) indexing — selects rows 1 and 2, may copy, and generalizes to non-contiguous indices. For 2D, `x[1:3, :]` is two-dimensional slicing.

**Q2:** How do you extract the last token hidden state for each sequence in a batch?

**A2:** If `hidden` has shape `(batch, seq, dim)`, use `hidden[:, -1, :]`, yielding `(batch, dim)`. Negative index `-1` refers to the last position along the sequence dimension. For variable-length sequences with padding, index the last *valid* token per sequence using lengths — not always `-1`.

**Q3:** What is boolean masking used for in transformers?

**A3:** Padding masks ignore pad tokens in attention and loss. Causal masks prevent attending to future tokens in decoder models. Boolean tensors mark which positions to keep (`True`) or mask out (`False`). Applied via `masked_fill` with `-inf` before softmax or by zeroing attention weights.

**Q4:** Does slicing create a copy?

**A4:** Often a **view** sharing storage — not guaranteed in all cases. If you modify a slice in-place, the parent tensor may change. Use `.clone()` for an independent tensor. Fancy indexing and boolean masking typically return new tensors.

**Q5:** How does `logits[torch.arange(N), labels]` work?

**A5:** Advanced indexing: for each row `i`, pick column `labels[i]`. `torch.arange(N)` selects all rows 0..N-1; `labels` picks one column per row. Result is a 1D tensor of length `N` — the logit score assigned to each sample's true class. Used in manual cross-entropy implementations.

### Indexing style guide

| Goal | Preferred pattern |
|------|-------------------|
| One row / column | `x[i]` or `x[:, j]` |
| Submatrix | `x[r0:r1, c0:c1]` |
| Filter by condition (analysis) | `x[mask]` |
| Filter keeping shape (training) | `torch.where(mask, x, fill)` |
| Per-row class logit | `x[arange(B), labels]` |
| Differentiable gather | `x.gather(dim, index)` |

Memorizing this table speeds up reading unfamiliar PyTorch model code.

**Q6:** When should you prefer `gather` over bracket indexing?

**A6:** Use `gather` when you need an operation that exports cleanly to TorchScript/ONNX or when gathering along one dimension with a known index tensor. Bracket indexing is fine for research notebooks; production pipelines sometimes standardize on `gather` and `index_select` for stability across PyTorch versions.

Bracket indexing is the fastest way to probe tensors in a debugger; `gather` is the safer choice when the same selection must run inside a compiled training step millions of times.

---

## 12. Summary

### Key notation

| Expression | Meaning |
|------------|---------|
| `x[i]` | Index dimension 0 |
| `x[:, j]` | All of dim 0, index j on dim 1 |
| `x[a:b]` | Slice dim 0 from a to b-1 |
| `x[x > 0]` | Boolean mask |
| `x[indices]` | Fancy indexing |

### Key terminology

- **Slice** — range selection with `:`
- **View** — shared-memory slice
- **Fancy indexing** — integer tensor indices
- **Boolean mask** — conditional selection
- **Causal mask** — prevent future token attention

---

## 13. Preview

Indexing selects existing elements. **Concatenating** joins tensors along an axis — stacking batch pieces, merging heads, or building sequences from segments. Next chapter covers `torch.cat`, `torch.stack`, and when to use each.

**Next chapter:** [Concatenating Tensors](06-concatenating-tensors.md)

---

## Lab

Companion notebook: [`app/pytorch/05_indexing_and_slicing.ipynb`](../../app/pytorch/05_indexing_and_slicing.ipynb)
