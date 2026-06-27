# Reshaping Tensors

## 1. Introduction

Tensors arrive in one shape and models demand another. A batch of images might be `(64, 3, 224, 224)` but a linear layer expects `(64, 150528)`. Multi-head attention splits `(batch, seq, dim)` into `(batch, heads, seq, head_dim)`. Reshaping changes **how you interpret** the same numbers without copying data (when possible).

This chapter covers `reshape`, `view`, `flatten`, `unsqueeze`, and `squeeze` — the tools that bridge between layers with incompatible shape expectations. Getting reshape wrong is a silent bug: same numel, wrong semantics, garbage outputs.

After this chapter you will be able to:

- Reshape tensors while preserving total element count.
- Add and remove size-1 dimensions with `unsqueeze` and `squeeze`.
- Choose between `view` and `reshape` safely.
- Split and merge dimensions for multi-head attention patterns.

**Where this appears in AI:** Flatten before the first linear layer in CNNs. Split embedding dim into heads in transformers. Add batch dimension for single-example inference. Reshape logits for cross-entropy loss.

Reshape errors rarely crash training on the first batch — they often produce silently wrong shapes that propagate until a later layer fails. Building the habit of printing `.shape` after every reshape saves hours of debugging.

The `-1` wildcard is convenient but hides arithmetic: if you reshape `(32, 768)` to `(32, 12, -1)` you implicitly require `768 / 12 = 64` per head. If hidden size is not divisible by head count, PyTorch raises an error — a useful guardrail in transformer configuration.

---

## 2. Intuition

> 💡 Intuition
>
> Reshape is **re-labeling the same box of numbers**. Imagine 12 eggs in a carton: 2 rows × 6 columns, or 3 rows × 4 columns, or one long line of 12. The eggs do not change — only how you count them. The rule: total count must stay the same.

```
  12 elements as (2, 6):        Same 12 as (3, 4):

  [ 0  1  2  3  4  5 ]         [ 0  1  2  3 ]
  [ 6  7  8  9 10 11 ]         [ 4  5  6  7 ]
                                [ 8  9 10 11 ]
```

**unsqueeze** inserts a dimension of size 1 — like putting one egg in its own tiny box. **squeeze** removes size-1 dimensions — collapsing empty packaging.

In transformers, hidden size 768 with 12 heads becomes 12 groups of 64: reshape `(batch, seq, 768)` → `(batch, seq, 12, 64)` → `(batch, 12, seq, 64)` for parallel head computation.

> 🔬 Deep Dive
>
> `view` requires contiguous storage; `reshape` may copy if needed. After `transpose`, prefer `reshape` or call `.contiguous().view()`. Both return a tensor viewing the same data when layout allows.

---

## 3. Formal Definitions

### Reshape invariant

Given tensor with numel \(N\), new shape \((d_0, \ldots, d_{k-1})\) must satisfy:

\[
d_0 \times d_1 \times \cdots \times d_{k-1} = N
\]

Use `-1` for one inferred dimension: `reshape(2, -1)` lets PyTorch compute the missing size.

### unsqueeze

Insert a dimension of size 1 at position `dim`:

\[
\text{shape } (a, b) \xrightarrow{\text{unsqueeze}(0)} (1, a, b)
\]

### squeeze

Remove all dimensions (or one specified dim) of size 1:

\[
\text{shape } (1, a, 1, b) \xrightarrow{\text{squeeze}} (a, b)
\]

---

## 4. Programming Perspective

```python
import torch

x = torch.arange(12).reshape(2, 2, 3)
print("original:", x.shape)       # torch.Size([2, 2, 3])

flat = x.flatten()
print("flatten:", flat.shape)     # torch.Size([12])

x4 = x.reshape(2, 3, 2)
print("reshape (2,3,2):", x4.shape)

# unsqueeze / squeeze
y = torch.tensor([1, 2, 3])       # (3,)
row = y.unsqueeze(0)                # (1, 3) — batch of one row
col = y.unsqueeze(1)                # (3, 1) — column vector
print(row.shape, col.shape)

z = torch.tensor([[[1], [2]]])    # (1, 2, 1)
print(z.squeeze().shape)          # (2,)
```

```python
# view vs reshape after transpose
t = torch.arange(12).reshape(3, 4).T  # non-contiguous
# t.view(12)  # may error
t_safe = t.reshape(12)  # works — copies if needed
print(t_safe.shape)
```

### Multi-head split pattern

```python
batch, seq, dim, heads = 2, 10, 768, 12
head_dim = dim // heads  # 64

x = torch.randn(batch, seq, dim)
# Split last dim into (heads, head_dim)
x_heads = x.reshape(batch, seq, heads, head_dim)
# Move heads before seq for batched matmul
x_heads = x_heads.permute(0, 2, 1, 3)  # (batch, heads, seq, head_dim)
print(x_heads.shape)
```

---

## 5. Visualizations

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

x = torch.arange(12).float().reshape(3, 4)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
shapes = [(3, 4), (2, 6), (4, 3)]
titles = ["(3, 4)", "(2, 6)", "(4, 3)"]

for ax, shape, title in zip(axes, shapes, titles):
    m = x.reshape(shape).numpy()
    im = ax.imshow(m, cmap="Blues", aspect="auto")
    ax.set_title(f"reshape {title}")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{int(m[i,j])}", ha="center", va="center")
    ax.set_xlabel("col")
    ax.set_ylabel("row")

plt.suptitle("Same 12 elements, different shapes")
plt.tight_layout()
plt.show()
```

**How to read:** Values 0–11 appear in different grid layouts. Row-major order is preserved: flatten then reshape reads elements in the same sequence. Changing shape changes which indices are "neighbors" — critical when reshaping weight matrices incorrectly.

```python
# Show effect of unsqueeze on dimensions
v = torch.tensor([1., 2., 3.])
print("1D:", v.shape)
print("unsqueeze(0):", v.unsqueeze(0).shape)  # (1, 3)
print("unsqueeze(1):", v.unsqueeze(1).shape)  # (3, 1)
```

---

## 6. Worked Examples

### Example 1: Flatten for linear layer

CNN output `(32, 256, 7, 7)` → linear needs `(32, 256*7*7)`.

```python
features = torch.randn(32, 256, 7, 7)
batch = features.shape[0]
flat = features.reshape(batch, -1)
print(flat.shape)  # torch.Size([32, 12544])
```

### Example 2: Reshape (4, 6) to (2, 3, 4)

```python
x = torch.arange(24).reshape(4, 6)
y = x.reshape(2, 3, 4)
print(x.shape, "->", y.shape)
assert x.numel() == y.numel()
```

### Example 3: Add batch dimension for inference

```python
single_image = torch.randn(3, 224, 224)  # one RGB image
batched = single_image.unsqueeze(0)       # (1, 3, 224, 224)
print(batched.shape)
```

### Example 4: Invalid reshape

```python
x = torch.arange(10)
try:
    x.reshape(3, 4)  # 12 != 10
except RuntimeError as e:
    print("Error:", e)
```

### Example 5: Multi-head reshape pipeline

Split embedding dimension into heads without mixing information:

```python
import torch

B, T, D = 2, 8, 64
num_heads, d_k = 8, 8
x = torch.randn(B, T, D)
# (B, T, D) -> (B, T, heads, d_k) -> (B, heads, T, d_k)
x = x.reshape(B, T, num_heads, d_k).transpose(1, 2)
print(x.shape)  # (2, 8, 8, 8)
```

The reshape groups features per head; the transpose places `heads` before `seq` for batched attention kernels.

### Example 6: Token classification flatten

```python
logits = torch.randn(4, 128, 50)   # batch, seq, classes
labels = torch.randint(0, 50, (4, 128))
logits_flat = logits.reshape(-1, 50)
labels_flat = labels.reshape(-1)
print(logits_flat.shape, labels_flat.shape)  # (512, 50), (512,)
```

Named entity recognition and token tagging apply loss per token by merging batch and sequence axes.

### Example 7: squeeze batch dimension after inference

```python
out = torch.randn(1, 10)  # batch=1 result
single = out.squeeze(0)
print(single.shape)  # (10,)
```

APIs returning one prediction often squeeze the batch axis before converting to Python lists.

### Example 8: view(-1) vs flatten — batch safety

```python
batch = torch.randn(16, 32, 32)
safe = batch.flatten(1)      # (16, 1024) — keeps batch
risky = batch.view(-1)       # (16384,) — loses batch structure
print(safe.shape, risky.shape)
```

Always prefer `flatten(start_dim=1)` when the leading dimension is batch.

### Example 9: Verify numel before reshape in a helper

```python
def safe_reshape(x, *shape):
    target = list(shape)
    if -1 in target:
        idx = target.index(-1)
        known = 1
        for i, d in enumerate(target):
            if i != idx and d != -1:
                known *= d
        target[idx] = x.numel() // known
    if x.numel() != int(torch.tensor(target).prod()):
        raise ValueError(f"Cannot reshape {tuple(x.shape)} to {tuple(target)}")
    return x.reshape(*target)

t = torch.randn(4, 12, 64)
out = safe_reshape(t, 4, 12, 8, 8)
print(out.shape)
```

Defensive helpers like this catch configuration mistakes early in notebook experiments.

### Example 10: CNN flatten with named dimensions

```python
import torch

x = torch.randn(8, 64, 14, 14)  # NCHW
n = x.size(0)
flat = x.reshape(n, -1)
print(flat.shape)  # (8, 12544)
linear_in = torch.nn.Linear(12544, 10)
logits = linear_in(flat)
print(logits.shape)
```

This is the classic AlexNet/VGG pattern: conv layers preserve spatial structure; the first fully connected layer requires explicit flatten.

---

## 7. AI Connection

> 🧠 AI Insight
>
> Cross-entropy loss in PyTorch expects logits of shape `(N, C)` and class indices of shape `(N,)`. If your model outputs `(N, seq, C)` for token classification, you reshape to `(N*seq, C)` and labels to `(N*seq,)` before calling `F.cross_entropy`.

**CNN → classifier:** `x.view(batch_size, -1)` or `flatten(1)` before `nn.Linear`.

**Multi-head attention:** Reshape hidden states into `(batch, heads, seq, head_dim)` so each head runs independent attention.

**Diffusion models:** Reshape noise tensors to match U-Net spatial dimensions `(B, C, H, W)`.

**Variable batch sizes:** Last batch may be smaller; reshape with `-1` only on safe dimensions.

**Export / ONNX:** Static reshape ops must match traced shapes; dynamic axes need explicit handling.

**Memory layout:** Reshape that only changes metadata is free; reshape that triggers a copy costs time and memory. When profiling slow layers, check whether unnecessary `.contiguous()` calls follow transpose+reshape chains.

**Head splitting in production code:** Frameworks like Hugging Face hide reshape permutes inside attention modules. When porting weights between implementations, mismatched head reshape order is a common source of silent quality degradation — always verify tensor layouts against a reference forward pass.

---

## 8. Common Mistakes

> ⚠️ Common Mistake
>
> **Reshape with wrong numel.** `(10,)` cannot become `(3, 4)`. Always check `tensor.numel()` equals product of new shape.

> ⚠️ Common Mistake
>
> **Using `view` after transpose without `contiguous()`.** Prefer `reshape` for clarity, or `.contiguous().view()`.

> ⚠️ Common Mistake
>
> **Confusing `flatten()` with `squeeze()`.** `flatten` merges dimensions into one; `squeeze` only removes size-1 axes. `(1, 3, 1, 3).squeeze()` → `(3, 3)` but shape depends on which dims are 1.

> ⚠️ Common Mistake
>
> **Reshaping breaks spatial structure in images.** `(B, 3, 224, 224)` flattened blindly loses which pixel was where — fine before a fully connected layer, disastrous if you meant to keep 2D structure for a conv layer.

**Correct understanding:** Numel is invariant. Document what each dimension means after reshape. Use `-1` for one inferred dim only.

When reviewing pull requests that touch tensor shapes, ask two questions: does the product of dimensions match numel, and does each axis still mean what the next layer expects? Reshape PRs that pass tests on square batches sometimes fail on odd batch sizes — always test `batch=1` and `batch=17`.

---

## 9. Exercises

### Easy

1. Create `torch.arange(20)`. Reshape to `(4, 5)` and `(5, 4)`. Print both.
2. Start with shape `(3, 4)`. Use `unsqueeze(0)` and `unsqueeze(2)`. Print shapes.
3. `torch.tensor([[[1], [2], [3]]])` — what does `squeeze()` produce?

### Medium

4. Reshape `(4, 6)` to `(2, 3, 4)` without changing element order. Verify first and last elements match before/after flatten.
5. Simulate CNN flatten: `(16, 512, 7, 7)` → `(16, 25088)`.
6. Split `(2, 10, 768)` into `(2, 10, 12, 64)` for 12 heads.
7. Explain when `reshape` copies data vs when it returns a view.

### Hard

8. Implement multi-head reshape + permute to get `(batch, heads, seq, head_dim)` from `(batch, seq, dim)`.
9. Reshape logits `(8, 20, 1000)` and labels `(8, 20)` for cross-entropy on 160 tokens.
10. Why does `x.reshape(-1)` always work but `x.reshape(3, -1)` might fail?

### Challenge

11. **Shape debugger:** Function that takes tensor and target shape, validates numel, prints friendly error if invalid, else reshapes.
12. **Head visualizer:** Random `(1, 8, 768)`, split into 12 heads, plot one head's 8×64 heatmap.

---

## 10. Mini Project

### Reshape Pipeline

Simulate a tiny vision + transformer pipeline with explicit reshape steps:

```python
import torch

# "CNN" output
cnn_out = torch.randn(8, 256, 4, 4)
print("1. CNN out:", cnn_out.shape)

# Flatten for FC (optional path)
flat = cnn_out.reshape(8, -1)
print("2. Flattened:", flat.shape)

# "Transformer" embedding
seq, dim, heads = 16, 768, 12
emb = torch.randn(8, seq, dim)
head_dim = dim // heads
heads_t = emb.reshape(8, seq, heads, head_dim).permute(0, 2, 1, 3)
print("3. Multi-head:", heads_t.shape)

# Restore merged dim
merged = heads_t.permute(0, 2, 1, 3).reshape(8, seq, dim)
print("4. Merged back:", merged.shape)
assert torch.allclose(emb, merged)
```

<details>
<summary>Mini project checklist</summary>

- [ ] At least 4 reshape steps documented
- [ ] Multi-head split and merge
- [ ] Assert round-trip equality

</details>

---

## 11. Interview Questions

**Q1:** What is the difference between `view` and `reshape`?

**A1:** Both change shape without changing data (when possible). `view` requires the tensor to be contiguous in memory; it fails otherwise. `reshape` returns a view if contiguous, otherwise makes a copy. After operations like `transpose`, use `reshape` for safety.

**Q2:** When can you use `-1` in reshape?

**A2:** Exactly one dimension can be `-1`; PyTorch infers it from numel and the other dimensions. Example: `x.reshape(2, -1)` on a 10-element tensor fails (10 not divisible by 2 evenly for integer shape — actually 10/2=5, works). On 12 elements, `(2, -1)` → `(2, 6)`.

**Q3:** How do you add a batch dimension to a single example?

**A3:** `x.unsqueeze(0)` if `x` is `(C, H, W)`, yielding `(1, C, H, W)`. Alternatively `x[None, ...]` or `x.reshape(1, *x.shape)`. Models trained on batches expect a leading batch axis even for batch size 1.

**Q4:** Why is multi-head reshape `(batch, seq, dim)` → `(batch, seq, heads, head_dim)` → `(batch, heads, seq, head_dim)`?

**A4:** First reshape splits the embedding dimension into independent head subspaces without mixing data. Then `permute` moves `heads` before `seq` so batched matmul runs all heads in parallel: each head is `(batch, seq, head_dim)` stacked as `(batch, heads, seq, head_dim)`.

**Q5:** What happens if you reshape incorrectly but numel matches?

**A5:** PyTorch will not error — you get a tensor with valid shape but **wrong semantics**. Weights might align with wrong input features; image pixels scramble. Always verify against a known reference or invariant (e.g. round-trip reshape restores original).

### Reshape decision checklist

Before calling `reshape` in a model, ask:

1. Does `numel()` match before and after?
2. Is the batch dimension still axis 0?
3. Did a recent `transpose` make the tensor non-contiguous?
4. Does the new layout match what the **next layer's documentation** expects?
5. Can you `assert` shapes with descriptive messages in unit tests?

Following this checklist prevents the majority of silent reshape bugs in research codebases.

---

## 12. Summary

### Key formulas

| Op | Effect |
|----|--------|
| `reshape(s)` | New shape, same numel |
| `flatten()` | Collapse to 1D |
| `unsqueeze(d)` | Insert size-1 at dim `d` |
| `squeeze(d)` | Remove size-1 at dim `d` |
| `-1` in shape | One inferred dimension |

### Key terminology

- **view** — zero-copy reshape if contiguous
- **contiguous** — row-major layout in memory
- **numel** — total elements (must preserve on reshape)
- **head_dim** — dim per attention head after split

---

## 13. Preview

Reshape reorganizes elements; **indexing and slicing** selects subsets without changing overall structure. You will need slicing to grab the last token's hidden state, mask padded positions, and implement attention masks. Next chapter covers `[ ]`, `:`, and boolean indexing.

**Next chapter:** [Indexing and Slicing](05-indexing-and-slicing.md)

---

## Lab

Companion notebook: [`app/pytorch/04_reshaping_tensors.ipynb`](../../app/pytorch/04_reshaping_tensors.ipynb)
