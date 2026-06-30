# Vocabulary Roadmap

> Shorter intro chapter. Read this **before** or **alongside** your first math chapters.

## Learning Objectives

- Stop panicking when you see an unfamiliar AI or math term.
- Know which chapter teaches each term properly.
- Distinguish **core material** (must learn now) from **preview material** (optional until later).

## Content

### If you see a word you do not know, you are not behind

Early chapters mention names like **ReLU**, **sigmoid**, **MSE**, and **softmax** before you have learned neurons or probability. That is **intentional** — a map of where the journey goes, not a test of what you should already know.

When you hit an unknown word:

1. **Pause** — confusion is a signal to look up, not a signal you are stupid.
2. **Check this page** — find the term in the tables below.
3. **Read the one-line meaning** — enough to keep going.
4. **Skip the details** if the chapter marks it with `📌 Preview — optional for now`.
5. **Return** when you reach the "Learn properly in" chapter.

> 💡 Intuition
>
> Learning AI is like reading a novel that drops character names in chapter 1 before you meet them. You are allowed to think "I will learn who that is later" and keep reading. This roadmap is the cast list.

### Core vs preview

| Label | Meaning | Your job |
|-------|---------|----------|
| **Core** | Required to finish the current chapter | Learn it now; do Easy exercises |
| **Preview** | Named early for context only | Skim the one-liner; skip formulas |
| **Payoff** | Chapter where previews become real lessons | Reread previews here; they should click |

Math chapters **01–04** (Functions through Gradients) use many **preview** labels. **06 Probability** and **03 Neural Networks** are **payoff** chapters for losses and activations.

---

## Activations (nonlinear functions inside neurons)

| Term | One line | Learn properly in |
|------|----------|-------------------|
| **Activation function** | Nonlinear rule applied after a weighted sum | [Single Neuron](../03-neural-networks/01-single-neuron.md) |
| **ReLU** | `max(0, x)` — zero if negative, else pass through | [Single Neuron](../03-neural-networks/01-single-neuron.md) |
| **Sigmoid** | Squashes a number to (0, 1) — probability-like | [Single Neuron](../03-neural-networks/01-single-neuron.md) |
| **GELU** | Smooth activation used in many transformers | [Decoder-Only Transformer](../04-transformers/04-decoder-only-transformer.md) |
| **Pre-activation** | Raw score before activation (often called `z`) | [Single Neuron](../03-neural-networks/01-single-neuron.md) |
| **Softmax** | Turns scores into probabilities that sum to 1 | [Probability](../01-math/06-probability.md) |

---

## Losses and training

| Term | One line | Learn properly in |
|------|----------|-------------------|
| **Loss function** | Single number measuring how wrong the model is | [Gradients](../01-math/04-gradients.md) |
| **MSE** | Mean squared error — for predicting numbers | [Gradients](../01-math/04-gradients.md) |
| **Cross-entropy** | Loss for classification; punishes confident wrong guesses | [Probability](../01-math/06-probability.md) |
| **Gradient descent** | Update weights to reduce loss | [Gradients](../01-math/04-gradients.md) |
| **Learning rate** | Step size when updating weights | [Gradients](../01-math/04-gradients.md) |
| **Backpropagation** | Chain rule applied to compute all gradients | [Backpropagation](../03-neural-networks/03-backpropagation.md) |
| **`loss.backward()`** | PyTorch call that runs backpropagation | [Backpropagation](../03-neural-networks/03-backpropagation.md) |
| **Optimizer** | Algorithm that applies gradients to weights (e.g. Adam) | [Backpropagation](../03-neural-networks/03-backpropagation.md) |
| **Logits** | Raw scores before softmax | [Probability](../01-math/06-probability.md) |

---

## Model building blocks

| Term | One line | Learn properly in |
|------|----------|-------------------|
| **Neuron** | Weighted sum + bias + activation | [Single Neuron](../03-neural-networks/01-single-neuron.md) |
| **Layer** | Many neurons in parallel | [Building a Layer](../03-neural-networks/02-building-a-layer.md) |
| **Weight** | Learned slope on each input | [Single Neuron](../03-neural-networks/01-single-neuron.md) |
| **Bias** | Learned intercept added after weighted sum | [Single Neuron](../03-neural-networks/01-single-neuron.md) |
| **`nn.Linear`** | PyTorch layer = matrix multiply + bias | [Building a Layer](../03-neural-networks/02-building-a-layer.md) |
| **Embedding** | Vector representation of a token or category | [Vectors](../01-math/03-vectors.md) (preview), transformers (full) |
| **Batch** | Many examples processed together | [Creating Tensors](../02-pytorch/01-creating-tensors.md) |

---

## Math fundamentals

| Term | One line | Learn properly in |
|------|----------|-------------------|
| **Function** | Rule: one input → one output | [Functions](../01-math/01-functions.md) |
| **Derivative** | Rate of change; slope at a point | [Derivatives](../01-math/02-derivatives.md) |
| **Vector** | Ordered list of numbers | [Vectors](../01-math/03-vectors.md) |
| **Dot product** | Sum of products; measures alignment | [Vectors](../01-math/03-vectors.md) |
| **Gradient** | Vector of partial derivatives | [Gradients](../01-math/04-gradients.md) |
| **Matrix** | Grid of numbers; linear transformation | [Matrices](../01-math/05-matrices.md) |
| **Probability** | Likelihood from 0 to 1 | [Probability](../01-math/06-probability.md) |
| **Chain rule** | Derivative of composed functions | [Derivatives](../01-math/02-derivatives.md) (preview), [Backpropagation](../03-neural-networks/03-backpropagation.md) |

---

## PyTorch and tensors

| Term | One line | Learn properly in |
|------|----------|-------------------|
| **Tensor** | Multi-dimensional array; core PyTorch object | [Creating Tensors](../02-pytorch/01-creating-tensors.md) |
| **Shape** | Size along each dimension, e.g. `(32, 784)` | [Creating Tensors](../02-pytorch/01-creating-tensors.md) |
| **`dtype`** | Number type (float32, int64, etc.) | [Creating Tensors](../02-pytorch/01-creating-tensors.md) |
| **`device`** | CPU or GPU where tensor lives | [Creating Tensors](../02-pytorch/01-creating-tensors.md) |
| **`requires_grad`** | Track this tensor for automatic derivatives | [Creating Tensors](../02-pytorch/01-creating-tensors.md) |
| **Matmul** | Matrix multiplication (`@`) | [Matrix Multiplication](../02-pytorch/02-matrix-multiplication.md) |
| **Transpose** | Swap rows and columns | [Transposing Tensors](../02-pytorch/03-transposing-tensors.md) |
| **Reshape / view** | Same data, different shape | [Reshaping Tensors](../02-pytorch/04-reshaping-tensors.md) |

---

## Transformers and attention

| Term | One line | Learn properly in |
|------|----------|-------------------|
| **Attention** | Weighted sum of values based on query–key match | [Attention Mechanism](../04-transformers/01-attention-mechanism.md) |
| **Self-attention** | Sequence attends to itself | [Self-Attention](../04-transformers/02-self-attention.md) |
| **Query (Q)** | What am I looking for? | [Attention Mechanism](../04-transformers/01-attention-mechanism.md) |
| **Key (K)** | What do I offer to match? | [Attention Mechanism](../04-transformers/01-attention-mechanism.md) |
| **Value (V)** | What information do I carry? | [Attention Mechanism](../04-transformers/01-attention-mechanism.md) |
| **Multi-head attention** | Several attentions in parallel | [Multi-Head Attention](../04-transformers/03-multi-head-attention.md) |
| **Causal mask** | Hide future tokens when predicting | [Decoder-Only Transformer](../04-transformers/04-decoder-only-transformer.md) |
| **Transformer block** | Attention + feed-forward + residuals | [Decoder-Only Transformer](../04-transformers/04-decoder-only-transformer.md) |
| **LLM** | Large language model; decoder-only transformer at scale | [Decoder-Only Transformer](../04-transformers/04-decoder-only-transformer.md) |

---

## Classical ML names (may appear early)

| Term | One line | Learn properly in |
|------|----------|-------------------|
| **Linear regression** | Fit a line \(y = wx + b\) | [Functions](../01-math/01-functions.md) (preview) |
| **Logistic regression** | Linear output + sigmoid for classification | [Single Neuron](../03-neural-networks/01-single-neuron.md) |
| **Neural network** | Composed functions with learned weights | [Functions](../01-math/01-functions.md) (preview), [Single Neuron](../03-neural-networks/01-single-neuron.md) |

---

## School math bridge

| Term | One line | Learn properly in |
|------|----------|-------------------|
| **PEMDAS** | Order of operations: parentheses, exponents, multiply/divide, add/subtract | [Math Basics](05-math-basics.md) |
| **Exponent** | `x ** n` — multiply x by itself n times | [Math Basics](05-math-basics.md) |
| **\(e\), exp** | Constant ≈ 2.718; `np.exp(x)` is \(e^x\) | [Math Basics](05-math-basics.md) (preview), [Probability](../01-math/06-probability.md) |
| **log** | Inverse of exp; used in cross-entropy | [Math Basics](05-math-basics.md) (preview), [Probability](../01-math/06-probability.md) |
| **sin, cos** | Wavy functions; positional encodings | [Math Basics](05-math-basics.md) (preview), [Special Tensors](../02-pytorch/07-special-tensors.md) |

---

## Part II preview (stubs — learn after Part I)

| Term | One line | Learn properly in |
|------|----------|-------------------|
| **Diffusion model** | Learn to denoise random noise into data | [Diffusion Models](../05-future/01-diffusion-models.md) |
| **Reinforcement learning** | Agent learns from rewards in an environment | [Reinforcement Learning](../05-future/02-reinforcement-learning.md) |
| **Policy gradient** | Train a policy by following reward signals | [Reinforcement Learning](../05-future/02-reinforcement-learning.md) |
| **Fine-tuning** | Adapt a pretrained model to a new task | Part I Transformers + future LLM depth |
| **RLHF** | Align models with human feedback | Part II (future) |

---

> ⚠️ Common Mistake
>
> Thinking you must understand every term in chapter 1 before moving on. Chapter 1 teaches **functions**. ReLU and sigmoid are **labels on the map**. Skip preview boxes and return when the roadmap says.

> 🧠 AI Insight
>
> Professional ML engineers still look up API details and paper notation. What separates them is knowing **which chapter of ideas** they are in — loss, optimization, architecture — not memorizing every formula on first sight.

> 🔬 Deep Dive
>
> Preview boxes use the format `📌 Preview — optional for now` with a link to the teaching chapter. If you are rewriting chapters, every ML term before its teaching chapter must use that format — see `.cursor/rules/handbook-textbook.mdc`.

## Summary

- Unknown words in early chapters are **previews**, not proof you are behind.
- Use the tables above: one-line meaning + where to learn properly.
- Math 01–04: focus on core math; skip preview details.
- Payoff chapters: Probability (losses), Single Neuron (activations), Attention chapters (transformers).
- Keep this page bookmarked while reading Part I.

## Further Reading

- [How to Use This Handbook](01-how-to-use-this-handbook.md) — what to do when you feel lost
- [Prerequisites](02-prerequisites.md) — who this book is for
- [Learning Path](03-learning-path.md) — full curriculum arc
- [Functions cheatsheet](../01-math/01-functions-cheatsheet.md) — first chapter review
