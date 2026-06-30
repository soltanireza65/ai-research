# Vectors

## 1. Introduction

A single number tells you one thing: a temperature, a score, a probability. But machine learning almost never works with just one number at a time. A word embedding is a list of 768 numbers. A pixel image is millions of values arranged spatially. A batch of training examples is a table of features.

The mathematical object for "a list of numbers with direction and magnitude" is a **vector**. Vectors are the atoms of linear algebra, and linear algebra is the language of neural networks, attention mechanisms, and GPU tensor operations.

After this chapter you will be able to:

- Represent data as vectors and interpret each component.
- Compute **magnitude** (length) with the norm \(\|v\|\).
- Add vectors and scale them — the operations behind weighted sums in attention.
- Understand the **dot product** as a measure of alignment (preview for similarity search and attention scores).
- Connect vectors to **embeddings**, weight rows, and activations in PyTorch.

**Where this appears in AI (preview):** Vectors appear in embeddings and attention — you will learn those words properly in [Vectors preview →](../00-intro/04-vocabulary-roadmap.md) later chapters. **Core now:** magnitude, addition, dot product as algebra.

> 📌 Preview — optional for now
>
> **Term:** embedding  
> **One line:** a vector representing a token or category  
> **Learn properly in:** [Vectors](03-vectors.md) (this chapter) + transformers later  
> You can skip until §7 if the word appeared too early.

**Suggested pacing (3 sessions):**

- Session A: §1–§3 + [cheatsheet](03-vectors-cheatsheet.md) skim
- Session B: §4–§6 + lab notebook
- Session C: Easy–Medium exercises + readiness checks in §12

---

## 2. Intuition

> 💡 Intuition
>
> A vector is an arrow. It starts at the origin (or anywhere, but we usually start at zero) and points to a location. The list `[3, 4]` means: walk 3 units east, 4 units north. The arrow encodes both **how far** (magnitude) and **which way** (direction).

```
        y
        |     * (3, 4)
        |    /|
        |   / |
        |  /  | 4  ← component along y
        | /   |
        +-----+---- x
          3
```

Two vectors with the same direction but different lengths are **parallel** but not equal. `[1, 2]` and `[2, 4]` point the same way; the second is twice as long.

**Vector addition** is "place the tail of the second arrow at the head of the first." The sum is the arrow from start to finish — like walking one displacement, then another.

```
  v1 ──────►
            └──► v2
  v1 + v2 ─────────►  (direct shortcut)
```

**Why AI cares:** A neural network layer computes weighted sums of input features. Each weight scales a component; the bias shifts the result. That is vector arithmetic in disguise.

> 🔬 Deep Dive
>
> Mathematicians distinguish **vectors** (fixed-length lists of numbers) from **scalars** (single numbers). In Python, `np.array([1, 2, 3])` is a vector; `3.14` is a scalar. Matrix rows and columns are vectors. A batch of 32 embeddings of dimension 768 is a **matrix** — 32 row vectors stacked together.

---

## 3. Formal Definitions

Every symbol is defined on first use.

### Vector

A **vector** in \(\mathbb{R}^n\) is an ordered list of \(n\) real numbers:

\[
\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}
\]

| Symbol | Meaning |
|--------|---------|
| \(\mathbf{v}\) | Bold or arrow notation for a vector (in code: an array) |
| \(v_i\) | The \(i\)-th **component** (entry) of the vector |
| \(n\) | **Dimension** — how many components |
| \(\mathbb{R}^n\) | "n-dimensional real space" — all vectors with \(n\) real components |

We often write \(\mathbf{v} = [v_1, v_2, \ldots, v_n]\) in row form for readability.

### Magnitude (norm)

The **Euclidean norm** (length) of \(\mathbf{v}\):

\[
\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}
\]

For a 2D vector \(\mathbf{v} = [3, 4]\):

\[
\|\mathbf{v}\| = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5
\]

This is the Pythagorean theorem: the length of the hypotenuse.

> **Plain English**
> Square each component, add them up, then take the square root — that is the vector's length.

> **Python**
> `np.linalg.norm(v)`

### Unit vector

A **unit vector** has length 1. Normalize by dividing by the norm:

\[
\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}
\]

The hat \(\hat{}\) means "unit version of."

### Vector addition and scalar multiplication

\[
\mathbf{u} + \mathbf{v} = [u_1 + v_1, u_2 + v_2, \ldots, u_n + v_n]
\]

\[
c \mathbf{v} = [c v_1, c v_2, \ldots, c v_n]
\]

Scalar \(c\) stretches or flips the vector. \(c = -1\) reverses direction.

> **Plain English**
> Add matching components one by one; multiply every component by the same scalar.

> **Python**
> `u + v` and `c * v`

### Dot product (preview)

For \(\mathbf{u}, \mathbf{v} \in \mathbb{R}^n\):

\[
\mathbf{u} \cdot \mathbf{v} = u_1 v_1 + u_2 v_2 + \cdots + u_n v_n = \sum_{i=1}^{n} u_i v_i
\]

Geometric meaning: \(\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta\) where \(\theta\) is the angle between them. Large positive dot product → vectors point similar directions. Zero → perpendicular (orthogonal). Negative → opposite directions.

> **Plain English**
> Multiply matching components and add the products — one number that measures how aligned two vectors are.

> **Python**
> `np.dot(u, v)` or `u @ v`

---

## 4. Programming Perspective

In Python, vectors are NumPy 1-D arrays (or PyTorch 1-D tensors).

| Mathematics | Python |
|-------------|--------|
| \(\mathbf{v} = [3, 4]\) | `v = np.array([3, 4])` |
| \(\|\mathbf{v}\|\) | `np.linalg.norm(v)` |
| \(\mathbf{u} + \mathbf{v}\) | `u + v` |
| \(2\mathbf{v}\) | `2 * v` |
| \(\mathbf{u} \cdot \mathbf{v}\) | `np.dot(u, v)` or `u @ v` |

```python
import numpy as np

v1 = np.array([3, 4])
v2 = np.array([-2, 1])

print("v1 =", v1)
print("v2 =", v2)
print("v1 + v2 =", v1 + v2)       # [1, 5]
print("2 * v1 =", 2 * v1)           # [6, 8]
print("||v1|| =", np.linalg.norm(v1))  # 5.0
print("v1 · v2 =", np.dot(v1, v2))     # -2
```

**Shape matters:** A vector has shape `(n,)`. A row vector with shape `(1, n)` and a column with shape `(n, 1)` behave differently in matrix multiplication — we cover that in the matrices chapter.

```python
# Normalizing to unit length
v = np.array([3.0, 4.0])
v_hat = v / np.linalg.norm(v)
print(v_hat)                        # [0.6, 0.8]
print(np.linalg.norm(v_hat))        # 1.0
```

Unit vectors appear when comparing directions without magnitude — cosine similarity uses normalized dot products.

---

## 5. Visualizations

Vector plots make direction and magnitude visible. Always label axes and interpret arrows.

```python
import numpy as np
import matplotlib.pyplot as plt

v1 = np.array([3, 4])
v2 = np.array([-2, 1])
v_sum = v1 + v2

fig, ax = plt.subplots(figsize=(7, 7))

origin = np.array([0, 0])
ax.quiver(*origin, v1[0], v1[1], angles="xy", scale_units="xy", scale=1,
          color="steelblue", label="v1 = [3, 4]")
ax.quiver(*origin, v2[0], v2[1], angles="xy", scale_units="xy", scale=1,
          color="coral", label="v2 = [-2, 1]")
ax.quiver(*origin, v_sum[0], v_sum[1], angles="xy", scale_units="xy", scale=1,
          color="seagreen", label="v1 + v2 = [1, 5]")

ax.set_xlim(-4, 5)
ax.set_ylim(-1, 6)
ax.set_aspect("equal")
ax.set_xlabel("x component")
ax.set_ylabel("y component")
ax.set_title("2D vectors and their sum")
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

**How to read this plot:**

- **Blue arrow** reaches (3, 4) — length 5.
- **Red arrow** reaches (-2, 1) — points left and up.
- **Green arrow** is the parallelogram shortcut: head-to-tail addition gives [1, 5].

```python
# Compare magnitudes and dot products
vectors = {
    "same direction": (np.array([1, 0]), np.array([3, 0])),
    "orthogonal": (np.array([1, 0]), np.array([0, 1])),
    "opposite": (np.array([1, 0]), np.array([-1, 0])),
}

for name, (u, v) in vectors.items():
    dot = np.dot(u, v)
    print(f"{name}: u·v = {dot}")
```

Output: same direction → positive (3); orthogonal → 0; opposite → negative (-1). This is why dot product measures alignment.

---

## 6. Worked Examples

### Example 1: Magnitude in 2D

Find \(\|[3, 4]\|\).

**Step 1:** Square each component: \(3^2 = 9\), \(4^2 = 16\).

**Step 2:** Sum: \(9 + 16 = 25\).

**Step 3:** Square root: \(\sqrt{25} = 5\).

### Example 2: Vector addition

\(\mathbf{u} = [1, 2, 3]\), \(\mathbf{v} = [4, -1, 0]\). Find \(\mathbf{u} + \mathbf{v}\).

**Step 1:** Add component-wise: \(1+4=5\), \(2+(-1)=1\), \(3+0=3\).

**Result:** \([5, 1, 3]\).

```python
u = np.array([1, 2, 3])
v = np.array([4, -1, 0])
print(u + v)  # [5 1 3]
```

### Example 3: Dot product

\(\mathbf{u} = [1, 2]\), \(\mathbf{v} = [3, 4]\). Find \(\mathbf{u} \cdot \mathbf{v}\).

**Step 1:** \(1 \times 3 = 3\)

**Step 2:** \(2 \times 4 = 8\)

**Step 3:** Sum: \(3 + 8 = 11\)

### Example 4: Cosine similarity (embedding preview)

Normalize both vectors, then dot product:

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

a = np.array([1.0, 2.0, 3.0])  # pretend embedding for "king"
b = np.array([1.1, 2.1, 2.9])  # pretend embedding for "queen"
c = np.array([-1.0, 0.0, 0.5]) # unrelated word

print(cosine_similarity(a, b))  # close to 1 — similar
print(cosine_similarity(a, c))  # lower — less similar
```

Search engines and RAG systems compare document embeddings this way.

### Example 5: Weighted sum (attention preview)

Attention output is a weighted sum of value vectors: \(\text{output} = \sum_i \alpha_i \mathbf{v}_i\) where weights \(\alpha_i\) sum to 1.

```python
values = np.array([
    [1.0, 0.0],  # value vector 1
    [0.0, 1.0],  # value vector 2
    [1.0, 1.0],  # value vector 3
])
weights = np.array([0.5, 0.3, 0.2])  # attention weights

output = np.sum(weights[:, None] * values, axis=0)
print(output)  # [0.7, 0.5]
```

Each value vector is scaled by its weight, then added — pure vector arithmetic.

---

## 7. AI Connection

> 🧠 AI Insight
>
> An embedding layer maps token ID 42 to a learned vector \(\mathbf{e}_{42} \in \mathbb{R}^{d}\). Similar words end up with similar vectors after training. The entire semantic geometry of a language model lives in vector space.

**Embeddings:** `nn.Embedding(vocab_size, d_model)` stores a matrix of shape `(vocab_size, d_model)`. Row \(i\) is the vector for token \(i\). Lookup is indexing: `embedding(token_id)` returns a vector.

**Linear layer as dot products:** A layer computes \(y_j = \mathbf{w}_j \cdot \mathbf{x} + b_j\) for each output neuron \(j\). Each weight row \(\mathbf{w}_j\) is a vector; the output is a stack of dot products between \(\mathbf{x}\) and each weight row.

**Attention scores:** \(\text{score}(i, j) = \mathbf{q}_i \cdot \mathbf{k}_j\) — dot product between query vector \(i\) and key vector \(j\). High score → attend more to position \(j\).

**Gradients:** The gradient of a loss with respect to all weights is a vector (actually organized as a matrix of the same shape as weights). Optimization adds a scaled negative gradient vector to the current weights.

**Batch dimension:** In practice you stack many vectors into matrices: `(batch_size, d_model)`. Each row is one example's vector. GPUs parallelize operations across rows.

---

## 8. Common Mistakes

> ⚠️ Common Mistake
>
> **Confusing a vector with a list of unrelated numbers.** Components must share a coordinate system. `[temperature, age, zipcode]` can be a feature vector, but only after scaling — raw zip codes are not commensurate with age without normalization.

> ⚠️ Common Mistake
>
> **Adding vectors of different dimensions.** `[1, 2] + [1, 2, 3]` is undefined. NumPy will broadcast in some cases and silently produce wrong results in others. Always check `.shape`.

> ⚠️ Common Mistake
>
> **Forgetting that dot product is a scalar, not a vector.** \(\mathbf{u} \cdot \mathbf{v}\) produces one number. Matrix multiplication generalizes this to many dot products at once.

> ⚠️ Common Mistake
>
> **Treating magnitude and direction as independent in embeddings.** Very large-magnitude vectors can dominate dot products. Many models normalize (layer norm) or use scaled dot products (divide by \(\sqrt{d}\) in attention) to keep values stable.

**Correct understanding:** Vectors encode direction and magnitude. Addition combines displacements. Dot product measures alignment. Embeddings, weights, and activations are all vectors or collections of vectors.

---

## 9. Exercises

### Easy

1. Compute \(\|[5, 12]\|\) by hand (hint: another Pythagorean triple).
2. If \(\mathbf{v} = [2, -1, 3]\), find \(2\mathbf{v}\) and \(\|\mathbf{v}\|\).
3. Plot vectors `[1, 0]` and `[0, 1]` from the origin. What is their dot product?
4. In Python, create `v1 = np.array([3, 4])` and `v2 = np.array([-2, 1])`. Print `v1 + v2` and `np.linalg.norm(v1)`.

### Medium

5. Find the unit vector in the direction of `[3, 4]`. Verify its length is 1.
6. Show that \(\mathbf{u} \cdot \mathbf{v} = 0\) for \(\mathbf{u} = [1, 2]\) and \(\mathbf{v} = [-2, 1]\). What does that imply about their angle?
7. Implement `cosine_similarity(a, b)` and test on three pairs of random 10-dimensional vectors.
8. Explain why word embeddings are high-dimensional (e.g., 768) rather than 2D — what tradeoff does dimensionality control?

### Hard

9. Three word vectors have \(\mathbf{a} \cdot \mathbf{b} = 8\), \(\|\mathbf{a}\| = 4\), \(\|\mathbf{b}\| = 4\). Find \(\cos\theta\) between them.
10. Given attention weights `[0.6, 0.3, 0.1]` and value vectors `[1,0]`, `[0,1]`, `[1,1]`, compute the weighted sum output by hand and in NumPy.
11. Plot 5 random 2D unit vectors as arrows. Which pair has the largest dot product?

### Challenge

12. **Embedding explorer:** Generate 8 random 2D vectors labeled "token_0" … "token_7". Plot them. Compute the full 8×8 cosine similarity matrix and print which pairs are most similar.
13. **Vector arithmetic analogy:** Implement \(\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}}\) with synthetic 5D vectors you design so the result is closest to \(\mathbf{v}_{\text{queen}}\). Explain what this analogy means geometrically.

---

## 10. Mini Project

### Vector Playground

Build a 2D vector playground that:

1. Lets you define at least 3 vectors.
2. Plots them as arrows from the origin with equal aspect ratio.
3. Shows the parallelogram for vector addition.
4. Prints magnitude, dot products between each pair, and cosine similarity.
5. Saves the figure to `book/assets/03-vectors-playground.png`.

```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

vectors = {
    "a": np.array([3, 4]),
    "b": np.array([-2, 1]),
    "c": np.array([1, 5]),
}

fig, ax = plt.subplots(figsize=(8, 8))
for name, v in vectors.items():
    ax.quiver(0, 0, v[0], v[1], angles="xy", scale_units="xy", scale=1, label=f"{name}={v}")
    print(f"||{name}|| = {np.linalg.norm(v):.3f}")

names = list(vectors.keys())
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        vi, vj = vectors[names[i]], vectors[names[j]]
        print(f"{names[i]}·{names[j]} = {np.dot(vi, vj)}")

ax.set_aspect("equal")
ax.legend()
ax.grid(True, alpha=0.3)
out = Path("book/assets/03-vectors-playground.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
```

<details>
<summary>Mini project checklist</summary>

- [ ] At least three labeled vectors plotted
- [ ] Magnitudes and pairwise dot products printed
- [ ] Equal aspect ratio on axes
- [ ] Figure saved to `book/assets/`

</details>

---

## 11. Interview Questions

**Q1:** What is a vector, and how does it differ from a scalar?

**A1:** A scalar is a single number. A vector is an ordered list of numbers — components that together describe magnitude and direction in a coordinate system. In ML, feature vectors represent one data point; embedding vectors represent tokens or entities. Scalars often appear as losses or learning rates; vectors appear as embeddings, activations, and gradients.

**Q2:** What does the dot product measure geometrically?

**A2:** The dot product \(\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta\). It is large and positive when vectors align (small angle), zero when perpendicular, and negative when pointing opposite directions. In attention, dot products between queries and keys produce scores — how much one position should attend to another. Cosine similarity normalizes magnitudes to compare pure direction.

**Q3:** How do embeddings use vectors in language models?

**A3:** Each token ID maps to a learned vector in \(\mathbb{R}^d\). During training, similar contexts push token vectors closer in space. The model reads and writes these vectors through every layer. Output logits are dot products between the final hidden state and an output weight matrix (vocabulary-sized). The geometry of embedding space encodes semantic relationships.

**Q4:** Why do we normalize vectors in some parts of neural networks?

**A4:** Normalization controls magnitude so that dot products and activations stay in a stable numeric range. Layer normalization rescales activations per token; attention divides by \(\sqrt{d_k}\) so dot products do not grow too large with dimension. Without normalization, training can explode or vanish — gradients become unusable.

**Q5:** What is the relationship between a matrix row and a vector?

**A5:** A matrix is a collection of vectors arranged as rows or columns. A weight matrix in `nn.Linear(in, out)` has `out` row vectors, each of dimension `in`. Matrix-vector multiplication computes the dot product of the input with each row — producing `out` scalar outputs stacked as a vector. This is how one linear layer transforms a vector into another vector.

---

## 12. Summary

### Key formulas

| Concept | Formula |
|---------|---------|
| Vector | \(\mathbf{v} = [v_1, v_2, \ldots, v_n]\) |
| Magnitude | \(\|\mathbf{v}\| = \sqrt{\sum_i v_i^2}\) |
| Addition | \(\mathbf{u} + \mathbf{v} = [u_1+v_1, \ldots, u_n+v_n]\) |
| Scalar multiply | \(c\mathbf{v} = [cv_1, \ldots, cv_n]\) |
| Dot product | \(\mathbf{u} \cdot \mathbf{v} = \sum_i u_i v_i\) |
| Cosine similarity | \(\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}\) |

### Key terminology

- **Vector** — ordered list of numbers with geometric direction
- **Scalar** — single number
- **Component** — one entry of a vector
- **Dimension** — number of components
- **Norm / magnitude** — length of a vector
- **Unit vector** — vector with norm 1
- **Dot product** — sum of products of corresponding components; measures alignment
- **Embedding** — learned vector representation of a discrete object (token, user, item)

### Readiness checks

Before the next chapter, you should be able to:

1. Compute \(\|[3, 4]\|\) by hand and verify with `np.linalg.norm`.
2. Add two 3D vectors component-wise and explain what vector addition means geometrically.
3. Compute a dot product by hand for 2D vectors and interpret the sign.
4. Normalize a vector to unit length in Python.
5. Explain why `u.shape` must match `v.shape` before adding.

If any item is shaky, reread §3 and the [cheatsheet](03-vectors-cheatsheet.md).

---

## 13. Preview

Vectors describe points and directions in space. The next chapter — **Gradients** — combines derivatives with vectors: when your function has many inputs, the **gradient** is a vector of partial derivatives pointing uphill. That vector drives gradient descent and `loss.backward()`.

After gradients, **Matrices** show how to apply linear transformations to many vectors at once — the operation inside every linear layer and attention block.

**Next chapter:** [Gradients](04-gradients.md)

---

## Lab

Companion notebook: [`app/math/03_vectors.ipynb`](../../app/math/03_vectors.ipynb)

## Review

- Cheatsheet: [Vectors — Cheatsheet](03-vectors-cheatsheet.md)
- Jargon: [Vocabulary Roadmap](../00-intro/04-vocabulary-roadmap.md)
