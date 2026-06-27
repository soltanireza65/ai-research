# Concatenating Tensors

## 1. Introduction

Models often build bigger tensors from smaller pieces: concatenate layer outputs, stack checkpoints, join encoder and decoder states, merge multi-scale features. **Concatenation** joins tensors **along an existing dimension** (growing that axis). **Stacking** creates a **new dimension** and places tensors as slices along it.

`torch.cat` and `torch.stack` are the two primitives. Confusing them produces wrong shapes — `(4, 6)` vs `(2, 3, 4)` from the same pair of `(2, 3)` tensors. This chapter teaches when to use each and how concatenation appears in ResNet skip connections, multi-head merge, and batch assembly.

After this chapter you will be able to:

- Concatenate tensors with `torch.cat` along any dimension.
- Stack tensors with `torch.stack` to introduce a new axis.
- Predict output shapes before calling cat/stack.
- Apply concatenation patterns in neural network architectures.

**Where this appears in AI:** ResNet concatenates and adds skip connections. Multi-head attention concatenates head outputs. Data loaders concatenate batches. Sequence models concatenate token embeddings with positional encodings (often via addition, but concat appears in fusion layers).

Understanding `cat` versus `stack` is like knowing whether you are extending a list in place or wrapping several lists inside a new container — the data may look similar on paper but the tensor rank tells a different story to every downstream layer.

In multi-GPU **pipeline parallelism**, activations from micro-batches are sometimes concatenated before the next stage receives them. Choosing the wrong dimension merges batch with sequence or splits heads incorrectly — errors that only appear at scale.

---

## 2. Intuition

> 💡 Intuition
>
> **cat** is **taping spreadsheets together** edge-to-edge along one side. Two `(2, 3)` tables cat along rows → `(4, 3)` (taller). Cat along columns → `(2, 6)` (wider). Same columns must align when taping vertically; same rows when taping horizontally.
>
> **stack** is **stacking papers in a new tray** — you add a dimension. Two `(2, 3)` sheets become one `(2, 2, 3)` block: dimension 0 indexes which sheet.

```
  cat dim=0 (vertical):     cat dim=1 (horizontal):

  [A]  (2×3)                [A | B]  (2×6)
  [B]  (2×3)

  stack dim=0:

  sheet 0: [A]  (2×3)
  sheet 1: [B]  (2×3)
  → shape (2, 2, 3)
```

In U-Net and similar architectures, **skip connections** concatenate encoder features with decoder features along the channel dimension — richer spatial detail meets upsampled semantics.

> 🔬 Deep Dive
>
> `torch.cat(tensors, dim=d)` requires all tensors to have the **same shape except on dimension d**. `torch.stack` requires **identical shapes** everywhere — it always adds one new dimension at position `dim`.

---

## 3. Formal Definitions

### cat (concatenate)

Given tensors \(T_1, \ldots, T_k\) each with shape matching on all dims except \(d\):

\[
\text{shape}(\text{cat}(T_1,\ldots,T_k, d))_j = \begin{cases}
\sum_i (T_i)_d & \text{if } j = d \\
(T_1)_j & \text{otherwise}
\end{cases}
\]

### stack

Given identical shapes \((s_0, \ldots, s_{m-1})\), stack at dim \(d\) inserts new dimension:

\[
\text{new shape} = (s_0, \ldots, s_{d-1}, k, s_d, \ldots, s_{m-1})
\]

where \(k\) is the number of tensors stacked.

---

## 4. Programming Perspective

```python
import torch

a = torch.ones(2, 3)
b = torch.zeros(2, 3)

cat_rows = torch.cat([a, b], dim=0)
print("cat dim=0:", cat_rows.shape)  # torch.Size([4, 3])

cat_cols = torch.cat([a, b], dim=1)
print("cat dim=1:", cat_cols.shape)  # torch.Size([2, 6])

stacked = torch.stack([a, b], dim=0)
print("stack dim=0:", stacked.shape)  # torch.Size([2, 2, 3])
```

```python
# Concatenate three vectors of length 4
v1 = torch.tensor([1., 2., 3., 4.])
v2 = torch.tensor([5., 6., 7., 8.])
v3 = torch.tensor([9., 10., 11., 12.])
combined = torch.cat([v1, v2, v3], dim=0)
print(combined)  # 12 elements
print(combined.shape)  # torch.Size([12])
```

```python
# Multi-head merge: concat heads on feature dim
batch, seq, heads, head_dim = 2, 8, 4, 16
heads_out = torch.randn(batch, seq, heads, head_dim)
merged = heads_out.reshape(batch, seq, heads * head_dim)
# equivalent to cat along last dim if split differently
print(merged.shape)  # (2, 8, 64)
```

### torch.cat vs torch.stack summary

| | `cat` | `stack` |
|---|-------|---------|
| Same shape required | Except on concat dim | Everywhere identical |
| New dimension? | No | Yes |
| Example | `(2,3)+(2,3)` → `(4,3)` at dim=0 | `(2,3)+(2,3)` → `(2,2,3)` at dim=0 |

---

## 5. Visualizations

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

a = torch.arange(6).reshape(2, 3).float()
b = torch.arange(6, 12).reshape(2, 3).float()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, M, title in zip(axes[:2], [a, b], ["a (2×3)", "b (2×3)"]):
    im = ax.imshow(M.numpy(), cmap="Blues")
    ax.set_title(title)
    for i in range(2):
        for j in range(3):
            ax.text(j, i, f"{int(M[i,j].item())}", ha="center", va="center")

cat01 = torch.cat([a, b], dim=0)
im = axes[2].imshow(cat01.numpy(), cmap="Blues")
axes[2].set_title("cat dim=0 → (4×3)")
for i in range(4):
    for j in range(3):
        axes[2].text(j, i, f"{int(cat01[i,j].item())}", ha="center", va="center")

plt.tight_layout()
plt.show()
```

**How to read:** Top two panels are inputs. Bottom-right is vertical concat — rows of `a` above rows of `b`. Values 0–5 then 6–11 top to bottom.

```python
# Visualize stack as separate layers
s = torch.stack([a, b], dim=0)
print("stack shape:", s.shape)
# s[0] is a, s[1] is b
```

---

## 6. Worked Examples

### Example 1: Concatenate along batch (dim 0)

```python
batch1 = torch.randn(16, 10)
batch2 = torch.randn(16, 10)
full = torch.cat([batch1, batch2], dim=0)
print(full.shape)  # torch.Size([32, 10])
```

### Example 2: Channel concat (dim 1 for NCHW)

```python
# Two feature maps same spatial size
f1 = torch.randn(8, 64, 32, 32)
f2 = torch.randn(8, 128, 32, 32)
fused = torch.cat([f1, f2], dim=1)
print(fused.shape)  # torch.Size([8, 192, 32, 32])
```

### Example 3: Stack for ensemble dimension

```python
pred_model_a = torch.randn(100, 10)
pred_model_b = torch.randn(100, 10)
ensemble = torch.stack([pred_model_a, pred_model_b], dim=0)
print(ensemble.shape)  # (2, 100, 10) — 2 models, 100 samples, 10 classes
mean_pred = ensemble.mean(dim=0)
```

### Example 4: Sequence concat

```python
prefix = torch.randn(2, 5, 768)  # system prompt tokens
suffix = torch.randn(2, 10, 768)  # user tokens
full_seq = torch.cat([prefix, suffix], dim=1)
print(full_seq.shape)  # (2, 15, 768)
```

### Example 5: Three vectors of length 4 (lab)

```python
v1 = torch.tensor([1., 2., 3., 4.])
v2 = torch.tensor([5., 6., 7., 8.])
v3 = torch.tensor([9., 10., 11., 12.])
vec = torch.cat([v1, v2, v3], dim=0)
print(vec, vec.shape)  # 12 elements
```

### Example 6: Split and re-cat round trip

```python
parts = torch.chunk(torch.randn(6, 4), 3, dim=0)
restored = torch.cat(parts, dim=0)
print(restored.shape)  # (6, 4)
```

`chunk` and `cat` are inverses — used when splitting batches across GPUs.

### Example 7: Stack dataset samples

```python
samples = [torch.randn(3, 32, 32) for _ in range(16)]
batch = torch.stack(samples, dim=0)
print(batch.shape)  # (16, 3, 32, 32)
```

Default collate behavior stacks numpy arrays into a batch tensor.

### Example 8: Wrong dim — doubled batch vs doubled sequence

```python
a = torch.randn(2, 5, 64)
b = torch.randn(2, 3, 64)
seq_cat = torch.cat([a, b], dim=1)    # (2, 8, 64) — correct for concat sequences
batch_cat = torch.cat([a, b], dim=0)  # (4, 5, 64) — WRONG if b was different seq len
print(seq_cat.shape)
```

Always verify which dimension is batch vs sequence before concatenating transformer tensors.

Document `cat` dimensions in function docstrings — future readers cannot infer whether you merged batch, time, or channels.

Concatenation is reversible with `split` when sizes are recorded.

### Example 9: Multi-head merge before output projection

```python
B, H, T, D = 2, 8, 10, 64
heads = torch.randn(B, H, T, D)
merged = heads.transpose(1, 2).reshape(B, T, H * D)
W_o = torch.randn(512, 512)
out = merged @ W_o
print(out.shape)  # (2, 10, 512)
```

Parallel heads concatenate along features, then `W_o` mixes them — the attention output block from the original Transformer paper.

---

## 7. AI Connection

> 🧠 AI Insight
>
> Multi-head attention runs parallel heads, then **concatenates** head outputs along the feature dimension before a final linear projection: `Concat(head_1, ..., head_h) @ W_O`.

**ResNet / DenseNet:** Skip connections add or concatenate feature maps — concat increases channel count, requiring a following conv to mix channels.

**Encoder-decoder:** Decoder inputs may concatenate previous hidden state with embedding of previous token (teacher forcing setups).

**Batch construction:** Distributed training gathers gradients; some pipelines `cat` micro-batch tensors.

**torch.vstack / hstack:** Convenience wrappers for 2D — know they map to cat on dim 0 or 1.

**Memory:** `cat` allocates new memory for the result — not in-place on inputs. Large cats can spike RAM during forward pass.

**Distributed training:** When gathering activations or gradients across GPUs, frameworks often `cat` or `stack` partial results along batch or shard dimensions. The concat dimension must match the partitioning strategy or reductions will silently combine wrong slices.

**torch.cat vs torch.vstack:** For 2D tensors, `vstack` is convenience sugar for `cat(..., dim=0)`. Prefer explicit `cat` with named `dim` in library code for clarity across arbitrary ranks.

**Gradient flow:** `cat` in forward splits gradients in backward — each input segment receives only its portion of `dL/doutput`. If one branch should not train, detach it before concatenating or mask gradients afterward.

---

## 8. Common Mistakes

> ⚠️ Common Mistake
>
> **Using `stack` when you mean `cat`.** Two `(32, 768)` batch tensors: `stack` → `(2, 32, 768)` (new dim for "which tensor"). `cat dim=0` → `(64, 768)` (one bigger batch). Completely different semantics.

> ⚠️ Common Mistake
>
> **Mismatched non-cat dimensions.** `cat([ (2,3), (2,4) ], dim=0)` fails — dim 1 differs. All dims except concat axis must match.

> ⚠️ Common Mistake
>
> **Wrong concat dimension for sequences.** Token concat is usually `dim=1` (seq) for `(batch, seq, dim)`. Cat on dim=2 joins features — rare for sequences.

> ⚠️ Common Mistake
>
> **Empty list to cat.** `torch.cat([])` errors. Guard against empty lists in dynamic pipelines.

**Correct understanding:** Sketch shapes. `cat` extends an axis; `stack` creates one. Match all non-concat dims exactly.

---

## 9. Exercises

### Easy

1. Cat two `(3, 4)` tensors on dim=0 and dim=1. Print shapes.
2. Stack three vectors length 5 on dim=0. What is the shape?
3. Cat three vectors length 4 into one length-12 vector (lab exercise).

### Medium

4. Given `(B, T1, D)` and `(B, T2, D)`, concatenate sequences to `(B, T1+T2, D)`.
5. Explain why `torch.cat([a, b], dim=0)` requires matching column counts for 2D tensors.
6. Merge 8 head outputs `(B, T, 64)` into `(B, T, 512)` via reshape/cat.
7. Compare `torch.vstack([a,b])` vs `torch.cat([a,b], dim=0)` for 2D inputs.

### Hard

8. Implement a simplified DenseNet block: concat input and conv output along channel dim.
9. Stack predictions from 5 fold models `(N, C)` into `(5, N, C)` and compute variance across models.
10. When does concatenation break autograd connection to one input? (Hint: it doesn't — gradients split.)

### Challenge

11. **Shape planner:** Function `plan_cat(tensors, dim)` validates compat shapes, prints result shape, returns cat or raises clear error.
12. **Multi-scale fusion:** Three feature maps `(B, 64, 32, 32)`, `(B, 64, 16, 16)`, `(B, 64, 8, 8)` — upsample smaller to 32×32, cat channels → `(B, 192, 32, 32)`.

---

## 10. Mini Project

### Tensor Joinery

Build a module that demonstrates cat/stack for sequences and features:

```python
import torch

def join_sequences(prefix, suffix):
    """(B, T1, D) + (B, T2, D) -> (B, T1+T2, D)"""
    assert prefix.shape[0] == suffix.shape[0]
    assert prefix.shape[2] == suffix.shape[2]
    return torch.cat([prefix, suffix], dim=1)

def stack_model_outputs(outputs):
    """List of (N, C) -> (M, N, C)"""
    return torch.stack(outputs, dim=0)

prefix = torch.randn(4, 5, 128)
suffix = torch.randn(4, 10, 128)
joined = join_sequences(prefix, suffix)
print("joined:", joined.shape)

outs = [torch.randn(100, 10) for _ in range(3)]
stacked = stack_model_outputs(outs)
print("stacked:", stacked.shape)
print("mean pred:", stacked.mean(0).shape)
```

<details>
<summary>Mini project checklist</summary>

- [ ] Sequence join on dim=1
- [ ] Model stack on new dim
- [ ] Shape assertions with clear messages

</details>

---

## 11. Interview Questions

**Q1:** What is the difference between `torch.cat` and `torch.stack`?

**A1:** `cat` joins tensors along an **existing** dimension, increasing that dimension's size. All other dimensions must match. `stack` creates a **new** dimension at the specified position; all input tensors must have identical shapes. `cat((2,3),(2,3), dim=0)` → `(4,3)`; `stack(..., dim=0)` → `(2,2,3)`.

**Q2:** How do you concatenate two batches of embeddings along the sequence dimension?

**A2:** If both have shape `(batch, seq, dim)` with same batch and dim, use `torch.cat([a, b], dim=1)` to get `(batch, seq_a + seq_b, dim)`. Ensure batch sizes and hidden dims match; only sequence length differs.

**Q3:** What shape constraint must hold for `torch.cat`?

**A3:** All tensors must have the same number of dimensions, and for every dimension except the concatenation dimension, sizes must be equal. On the concat dimension, sizes can differ — the output size is the sum of input sizes on that dim.

**Q4:** Where is concatenation used in multi-head attention?

**A4:** After computing attention in parallel for each head, outputs of shape `(batch, seq, head_dim)` per head are concatenated along the feature dimension to `(batch, seq, num_heads * head_dim)`, then multiplied by output projection matrix `W_O`.

**Q5:** Does `torch.cat` copy data?

**A5:** Yes — the result is a new tensor with contiguous storage holding copies of input data in order. Inputs remain unchanged. Gradients in backward propagate to each input segment separately. For very large tensors, memory planning matters.

### cat vs stack decision tree

```
Same shape tensors?
  ├─ NO → cannot stack; maybe cat if only concat dim differs
  └─ YES → need new axis?
        ├─ YES → stack (e.g. list of samples → batch)
        └─ NO → cat (e.g. extend sequence or channels)
```

When in doubt, write expected output shape on paper before choosing.

**Q6:** How is concatenation used in U-Net skip connections?

**A6:** Encoder feature maps at a given resolution are concatenated with upsampled decoder features along the channel dimension. This preserves fine spatial detail from the encoder while combining semantic information from the decoder. A following convolution mixes the doubled channel count back to the desired width.

The same concat pattern appears in multimodal models when text and image embeddings are joined along sequence or feature axes before a fusion transformer block runs.

---

## 12. Summary

### Key rules

| Operation | Shape change | Requirement |
|-----------|--------------|-------------|
| `cat(..., dim=d)` | dim d sums | other dims match |
| `stack(..., dim=d)` | new dim size = num tensors | all shapes identical |

### Key terminology

- **concatenate (cat)** — join along existing axis
- **stack** — new axis for list of tensors
- **skip connection** — concat or add earlier features
- **channel dim** — often dim 1 in NCHW for feature concat

---

## 13. Preview

You can create, multiply, transpose, reshape, slice, and join tensors. The last PyTorch fundamentals chapter covers **special tensors** — factories like `eye`, `arange`, `linspace`, and random generators that appear in initialization, positional encodings, and debugging every day.

**Next chapter:** [Special Tensors](07-special-tensors.md)

---

## Lab

Companion notebook: [`app/pytorch/06_concatenating_tensors.ipynb`](../../app/pytorch/06_concatenating_tensors.ipynb)
