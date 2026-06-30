# Learning Path

> Shorter template for `book/00-intro/` chapters. Target ≥800 words.

## Learning Objectives

- See the full 0→100 arc from math foundations through transformers and beyond.
- Understand why topics appear in this order — and what each module unlocks.
- Know which chapters are complete today versus planned stubs in Part II.
- Plan your study timeline and know when to revisit earlier material.

## Content

This handbook is a single continuous path, not a menu of unrelated tutorials. Each module exists because the **next** module depends on it. Skip the math, and PyTorch tensor operations feel like magic incantations. Skip PyTorch, and neural network code becomes copy-paste. Skip neural networks, and transformer attention is just a formula you memorized.

This chapter is the map. Use it when you wonder *why am I learning probability right now?* or *what comes after transformers?*

If unfamiliar words appear **before** they are explained (e.g. sigmoid in Functions), use the [Vocabulary Roadmap](04-vocabulary-roadmap.md). Early chapters use **preview** labels; payoff chapters (Probability, Single Neuron, Attention) teach those terms properly.

### The Arc at a Glance

```
  Part I — Foundations to Transformers (written)
  ─────────────────────────────────────────────
  00 Intro          → how to study, setup, vocabulary roadmap, math basics bridge
  01 Math           → language of change and uncertainty
  02 PyTorch        → executable linear algebra on GPU/CPU
  03 Neural Nets    → composable functions trained by gradients
  04 Transformers   → attention, self-attention, full decoder stack

  Part II — Research Horizons (stubs expanding)
  ─────────────────────────────────────────────
  05 Future         → diffusion, RL, reading papers, LLM depth
```

The goal of Part I is not to make you a Kaggle grandmaster. It is to make you someone who can **read a transformer paper**, **implement a small model from scratch in PyTorch**, and **debug training** because you understand loss landscapes and backpropagation — not because you found a Stack Overflow snippet.

> 💡 Intuition
>
> Imagine learning to repair modern electric vehicles. You could start by swapping pre-built battery modules — that is the "just use the API" path. Or you could learn voltage, current, and circuit diagrams first, then battery management systems, then the full drivetrain. This handbook takes the second path. It is slower at the beginning and dramatically more empowering when something breaks at 2 a.m. before a demo.

### Module 01 — Math Fundamentals

**Start here if school math is rusty:** [Math Basics](05-math-basics.md) (PEMDAS, exponents, axes, `e`/`log`/`sin` previews).

**Chapters:** Functions → Derivatives → Vectors → Gradients → Matrices → Probability

**Why first:** Every training algorithm asks "which direction improves the loss?" That question is calculus (derivatives, gradients) on vectors and matrices. Probability explains why models output distributions, why cross-entropy is a loss function, and why "confidence" is not certainty.

**What you unlock:** Reading loss formulas, understanding learning rate effects, seeing neural networks as composed functions, interpreting weight matrices as linear transformations.

| Chapter | AI payoff |
|---------|-----------|
| Functions | Core: input→output rules; previews only for ML names |
| Derivatives | Slope of loss, sensitivity, optimization steps |
| Vectors | Embeddings, feature vectors, weight rows/columns |
| Gradients | Multivariable optimization, direction of steepest descent |
| Matrices | Weight layers, batch operations, attention as matmul |
| Probability | Classification outputs, sampling, uncertainty |

Do not rush this module. It is the compression algorithm for everything that follows.

> ⚠️ Common Mistake
>
> Jumping to `04-transformers` because ChatGPT is hot. Attention is matrix multiplication plus softmax plus clever masking. Without matrices and gradients, you will memorize shapes (`[batch, heads, seq, dim]`) without knowing why those dimensions exist. The learning path is ordered by **dependency**, not **hype**.

### Module 02 — PyTorch Fundamentals

**Chapters:** Creating Tensors → Matrix Multiplication → Transposing → Reshaping → Indexing → Concatenating → Special Tensors

**Why here:** You now know what matrix multiplication *means* mathematically. PyTorch is how you *do* it on large arrays with automatic differentiation ready for training.

**What you unlock:** Reading any open-source model code, shaping batches correctly, understanding why a transpose fixes a dimension mismatch, debugging `RuntimeError: size mismatch` with confidence.

Each PyTorch chapter maps 1:1 to a lab notebook under `app/pytorch/`. Run every cell with the handbook open side-by-side.

> **Module review before Module 03:** Skim all seven PyTorch cheatsheets in `book/02-pytorch/` — shape rules, `@`, `.T`, `view`, indexing. If any cheatsheet feels foreign, rerun that lab before building layers.

> 🔬 Deep Dive
>
> PyTorch is not "the ML framework" in this curriculum — it is **executable notation**. When the handbook writes \(C = A B\), the lab writes `C = A @ B`. When backprop later needs \(\partial L / \partial W\), PyTorch's autograd tracks the same graph you would derive by hand — the handbook's backprop chapter connects those views explicitly.

### Module 03 — Neural Networks

**Chapters:** Single Neuron → Building a Layer → Backpropagation

**Why here:** A neuron is a tiny function: linear transform + nonlinearity. A layer stacks neurons. Training stacks layers and uses gradients from Module 01, implemented via PyTorch from Module 02.

**What you unlock:** Implementing a MLP from scratch, understanding why depth helps, deriving backprop on paper and trusting `loss.backward()` because you know what it computes.

Companion lab: `app/neural_networks/03_backpropagation.ipynb` — one-weight `backward()` and a tiny training loop. Budget extra time; this is the conceptual hinge of deep learning.

> 🧠 AI Insight
>
> Modern LLMs are neural networks with billions of parameters and transformer blocks instead of plain dense layers — but the **training loop** is unchanged: forward pass, loss, backward pass, optimizer step. Mastering the three-chapter neural network module means every giant model is familiar architecture plus scale plus tricks, not an alien species.

### Module 04 — Transformers

**Chapters:** Attention Mechanism → Self-Attention → Multi-Head Attention → Decoder-Only Transformer

**Why here:** Attention is the mechanism behind GPT-style models. You need matrices (queries, keys, values), softmax (probability module), and PyTorch reshaping (batch/head dimensions).

**What you unlock:** Reading the "Attention Is All You Need" paper with comprehension, implementing a tiny GPT-like model, understanding KV-cache and causal masking at a structural level.

This is the end of Part I's core narrative: from "what is a derivative?" to "how does a decoder-only transformer produce the next token?"

### Part II — Future Horizons (Stubs)

Part I gets you to **transformer literacy**. Part II extends toward **research literacy** — topics that assume Part I is solid:

| Stub chapter | Focus | Builds on |
|--------------|-------|-----------|
| [Diffusion Models](../05-future/01-diffusion-models.md) | Generative denoising, score matching, image/audio synthesis | Probability, gradients, neural nets |
| [Reinforcement Learning](../05-future/02-reinforcement-learning.md) | Policies, rewards, exploration vs exploitation | Probability, optimization, PyTorch loops |
| [Reading AI Papers](../05-future/03-reading-ai-papers.md) | How to parse abstracts, methods, ablations | All of Part I + notation fluency |

Large language models (LLMs) are not a separate stub chapter because **Module 04 already builds decoder-only transformers** — the architecture behind GPT. Part II will add depth on scaling, pretraining objectives, fine-tuning, and alignment as those chapters expand.

> 💡 Intuition
>
> Part I builds the engine and transmission. Part II covers specialized vehicles — image generators (diffusion), game-playing agents (RL), and the technical literature (papers). You cannot race if you have not assembled the drivetrain; you can assemble the drivetrain without ever racing. Both are valid goals, but this handbook sequences them in that order.

### Suggested Timelines

Rough estimates for a working engineer studying part-time (8–10 hrs/week):

| Phase | Duration | Milestone |
|-------|----------|-----------|
| Math (01) | 7–12 weeks | Can explain gradient descent on paper (includes Math Basics if needed) |
| PyTorch (02) | 3–5 weeks | Can debug tensor shapes in a small model |
| Neural Nets (03) | 2–4 weeks | Can implement backprop for a MLP |
| Transformers (04) | 3–5 weeks | Can explain self-attention to a colleague |
| Part II stubs | TBD | Choose depth by research interest |

Accelerate if full-time; extend if needed. **Completion beats calendar.**

### Spiraling and Review

The path is linear in the table of contents but **spiral** in practice:

- Re-read **Gradients** after **Backpropagation**.
- Re-read **Matrix Multiplication** after **Multi-Head Attention**.
- Re-run early labs after finishing Module 04 — you will change code you previously copied blindly.

Mark chapters complete in [`SUMMARY.md`](../SUMMARY.md) only when you can explain the core idea without the book open.

### What "100" Means Here

"0→100" in this handbook does not mean "ready to publish at NeurIPS tomorrow." It means:

- **0:** engineer who uses AI APIs but fears the math
- **100 (Part I complete):** engineer who can implement and train a small transformer, understand why it works, and read introductory papers
- **100+ (Part II):** engineer moving toward research — generative models, RL agents, systematic paper reading

The ceiling is intentionally high; the floor is intentionally low. You start at the floor.

## Summary

- The path is **Math → PyTorch → Neural Networks → Transformers → Part II (diffusion, RL, papers)**.
- Order follows **dependencies**, not trend cycles — skipping math undermines transformers.
- Part I is the main written curriculum; **Part II stubs** extend toward research topics.
- LLM architecture is covered in **Module 04**; Part II adds scaling and training depth later.
- Use [`SUMMARY.md`](../SUMMARY.md) to track progress; revisit earlier chapters after later ones click.

## Further Reading

- [How to Use This Handbook](01-how-to-use-this-handbook.md) — daily workflow
- [Prerequisites](02-prerequisites.md) — setup before Module 01
- [`book/SUMMARY.md`](../SUMMARY.md) — checkable chapter list
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — read after Module 04 (not before)
