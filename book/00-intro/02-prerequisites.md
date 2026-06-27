# Prerequisites

> Shorter template for `book/00-intro/` chapters. Target ≥800 words.

## Learning Objectives

- Confirm you have the Python skills assumed by the labs and handbook.
- Set up the project environment with `uv sync` and launch Jupyter with `just lab`.
- Reframe math anxiety as a normal starting point — no prior ML or calculus required.
- Know what you can defer learning until the handbook introduces it.

## Content

You do not need a machine learning PhD to start this handbook. You do not even need a machine learning *course*. What you need is comfort writing Python, curiosity about how modern AI systems work, and willingness to learn math **when it becomes necessary** — not years before.

This chapter tells you exactly what is assumed, how to install the tooling, and how to think about the math you may have avoided until now.

### Who This Handbook Is For

The ideal reader is an **experienced software engineer**:

- You have written Python for scripts, services, or data pipelines.
- You may have used AI APIs or imported scikit-learn once, but you do not deeply understand what happens inside a neural network.
- You have little or no formal background in calculus, linear algebra, or probability — or you learned it once and forgot it.

That profile is not a weakness. It is the **target audience** every chapter is written for. The handbook explains notation when it appears, maps formulas to Python, and connects each idea to AI applications. You are not expected to arrive already knowing what a Jacobian is.

> 💡 Intuition
>
> You already have the hardest prerequisite: you can program. Programming is the skill of taking a precise rule and making a machine follow it. Machine learning is the same activity with different rules — rules learned from data instead of typed by you. The math in this handbook is simply the language those rules are written in. You learned syntax for your first programming language; you can learn syntax for gradients and matrices the same way.

### Python Skills You Should Have

You do not need to be a Python core developer. You **do** need fluency with everyday patterns that appear in every lab notebook:

| Skill | Example | Where it appears |
|-------|---------|------------------|
| Variables and types | `x = 3.14`, `name: str = "relu"` | All notebooks |
| Functions | `def f(x): return x ** 2` | Math chapters, custom layers |
| Lists and loops | `for i in range(10): ...` | Plotting, training loops |
| List comprehensions | `[x**2 for x in xs]` | Compact NumPy-style thinking |
| Imports | `import numpy as np`, `import matplotlib.pyplot as plt` | Every lab |
| Basic classes (helpful) | `class LinearLayer:` | Neural network chapters |

If any of these feel rusty, spend an afternoon refreshing them before Chapter 1. You do not need async/await, metaclasses, or packaging expertise for this curriculum.

**Libraries introduced here** — you do not need prior experience:

- **NumPy** — arrays and numerical operations (taught alongside math chapters)
- **Matplotlib** — plotting functions and loss curves (taught with visualizations)
- **PyTorch** — tensors and autograd (dedicated module in Part I)

> ⚠️ Common Mistake
>
> Believing you must master PyTorch, NumPy, *and* advanced math before opening Chapter 1. That order is backwards. This handbook teaches tools in the same order you need them: functions and plots first, tensors after you understand vectors and matrices, PyTorch after you understand what a gradient means. Starting with a 40-hour PyTorch tutorial before reading `01-functions.md` creates confusion, not speed.

### Environment Setup

The project uses **[uv](https://docs.astral.sh/uv/)** for fast, reproducible Python environments and **[just](https://github.com/casey/just)** as a command runner. From the repository root:

```bash
uv sync
just lab
```

**What `uv sync` does:** reads `pyproject.toml`, creates or updates a virtual environment in `.venv/`, and installs dependencies including PyTorch, NumPy, Matplotlib, and Jupyter. Run it once when you clone the repo, and again whenever dependencies change.

**What `just lab` does:** starts JupyterLab pointed at the `app/` directory so notebook paths align with handbook modules (`app/math/`, `app/pytorch/`, etc.).

Other useful commands from the `justfile`:

```bash
just validate-book          # check handbook chapter quality
just test                   # run pytest
just lint                   # ruff check
```

If `just` is not installed globally, `uv sync` installs it as a dev dependency — invoke via `uv run just lab` if needed.

> 🔬 Deep Dive
>
> Python 3.14+ is specified in `pyproject.toml`. The project pins modern NumPy, Matplotlib, and PyTorch versions so lab code matches current APIs. If you use another Python manager (conda, pyenv), ensure you meet the version floor; otherwise stick to `uv sync` for the least friction. Jupyter kernels should use the `.venv` interpreter created by uv — in JupyterLab, verify the kernel name matches your project environment if cells fail with `ModuleNotFoundError`.

### What You Do *Not* Need

Explicitly **not** required before starting:

- Prior machine learning or deep learning courses
- Calculus, linear algebra, or probability coursework
- GPU hardware (CPU is fine for all Part I labs)
- Research paper reading experience
- LangChain or LLM application development (those dependencies exist for other project work, not Part I core labs)

The handbook **will teach** the math you need, in this order: functions → derivatives → vectors → gradients → matrices → probability, then PyTorch, then neural networks, then transformers. Trust the sequence.

> 🧠 AI Insight
>
> Industry hiring often lists "ML fundamentals" as a prerequisite, which pushes engineers toward framework tutorials. Framework tutorials teach *calls* — `model.fit`, `tensor.cuda`, `nn.Linear`. This handbook teaches *mechanisms* — what loss minimizes, why backprop chains derivatives, how attention computes weighted sums. Engineers who skip mechanisms can ship demos but struggle to debug training instability or read papers. Starting with math-for-AI intuition, even if it feels slower, compounds when you reach transformers and beyond.

### Math Anxiety: A Practical Reframe

Many capable engineers avoid AI because a past math class made them feel slow or stupid. If that is you, read this carefully: **the handbook is written for people who feel that way.**

Patterns that help:

1. **Intuition before symbols** — every chapter opens with plain-language models and diagrams before introducing \(f(x)\) or \(\nabla L\).
2. **Code as ground truth** — if an equation is confusing, the Python equivalent in the same chapter is often clearer. Run it with small numbers.
3. **Small steps** — Easy exercises exist so you can verify understanding before Medium and Hard tiers ask you to combine ideas.
4. **Revisiting is normal** — chapter 4 makes more sense after chapter 6. That is intentional spiraling, not failure.

You are not trying to become a mathematician. You are trying to become an engineer who **reads** mathematical ideas confidently enough to implement and debug AI systems. That is a smaller, achievable goal.

### Hardware and Time

A laptop with 8 GB RAM is sufficient for Part I. Training large models is not the focus — understanding small tensors, tiny networks, and attention on toy sequences is. Budget **5–15 hours per math chapter** and **3–8 hours per PyTorch chapter** depending on prior exposure; adjust downward if you are full-time on this, upward if you are squeezing evenings.

### Before You Begin Checklist

- [ ] Python 3.14+ available (via `uv sync`)
- [ ] `uv sync` completed without errors
- [ ] `just lab` opens JupyterLab with `app/` visible
- [ ] You read [How to Use This Handbook](01-how-to-use-this-handbook.md) for the two-layer workflow
- [ ] You skimmed [`SUMMARY.md`](../SUMMARY.md) to see the full arc

When all boxes are checked, open `book/01-math/01-functions.md` and its lab `app/math/01_functions.ipynb`.

## Summary

- **Assumed:** everyday Python fluency; **not assumed:** ML, calculus, or PyTorch experience.
- Run **`uv sync`** once to install dependencies; use **`just lab`** to open notebooks in `app/`.
- Math anxiety is common — the curriculum teaches intuition first and maps notation to code.
- No GPU or prior research background required for Part I.
- Complete the checklist above, then start with the Functions chapter in `01-math`.

## Further Reading

- [How to Use This Handbook](01-how-to-use-this-handbook.md) — study workflow and validation
- [Learning Path](03-learning-path.md) — where prerequisites lead
- [uv documentation](https://docs.astral.sh/uv/) — package and environment management
- [Python official tutorial](https://docs.python.org/3/tutorial/) — refresh language basics if needed
