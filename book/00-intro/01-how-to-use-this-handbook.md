# How to Use This Handbook

> Shorter template for `book/00-intro/` chapters. Target ≥800 words.

## Learning Objectives

- Explain the two-layer model: textbook chapters in `book/` paired with runnable labs in `app/`.
- Follow the recommended study workflow for each chapter.
- Know where Cursor rules and skills live when extending or rewriting chapters.
- Run `just validate-book` to check chapter quality before submitting changes.

## Content

Welcome to the **AI Research Handbook** — a structured path from software engineering to the mathematical foundations of modern AI. This repository is not a loose collection of notebooks. It is a **two-layer learning system** designed so you can read deeply, run code immediately, and connect theory to practice without switching contexts.

### The Two-Layer Model

Think of the handbook as a textbook sitting on your desk, and the app as the lab bench beside it.

| Layer | Location | What it is | Your job |
|-------|----------|------------|----------|
| **Handbook** | `book/<module>/` | Full textbook chapters: intuition, derivations, exercises, interview questions | Read, take notes, do exercises |
| **Labs** | `app/<module>/` | Jupyter notebooks aligned chapter-for-chapter | Run cells, change values, break things |
| **Utilities** | `app/utils/` | Shared plotting helpers used across labs | Import and reuse; read when curious |
| **Assets** | `book/assets/` | Saved figures from labs | Reference in your notes |

Every major chapter in `book/` has a companion notebook in `app/` with the same topic order. For example, `book/01-math/01-functions.md` pairs with `app/math/01_functions.ipynb`. The handbook explains *why*; the lab shows *how* in executable Python.

```
  ┌─────────────────────┐         ┌─────────────────────┐
  │   book/ chapter     │  read   │   app/ notebook     │
  │   (textbook prose)  │ ──────► │   (runnable code)   │
  └─────────────────────┘         └─────────────────────┘
            │                                 │
            └────────── experiment ◄──────────┘
                    (change, observe, break)
```

You do not need to choose one layer. The design assumes you use **both**, in sequence, for every topic.

> 💡 Intuition
>
> Treat each chapter like a lecture plus a lab section. Read the handbook first to build mental models — what is a gradient, why does attention use dot products — then open the notebook to make those models concrete. If you only read, the math stays abstract. If you only run cells, you risk becoming a "API tourist" who can call PyTorch but cannot explain what a tensor operation means. The two layers together prevent both failure modes.

### Recommended Study Workflow

For each chapter, follow this loop until the ideas feel natural:

1. **Read** the chapter markdown in `book/`. Do not skim the intuition sections; they exist because formal notation comes later.
2. **Run** the companion notebook via `just lab` (see [Prerequisites](02-prerequisites.md) for setup).
3. **Experiment** — change learning rates, swap activation functions, plot different ranges. Break things on purpose and observe what fails.
4. **Complete** the exercises at the end of the chapter. Start with Easy; do not skip to Hard until Medium feels routine.
5. **Check off** the chapter in [`SUMMARY.md`](../SUMMARY.md) so you can see progress across the full arc.

A single pass is rarely enough. Revisit chapters after you learn related material: derivatives make more sense after gradients; matrix multiplication clicks harder after you have built a layer. Spaced repetition is built into the curriculum — later parts explicitly reference earlier ones.

> ⚠️ Common Mistake
>
> Running notebook cells top-to-bottom once and moving on. That produces familiarity with outputs, not understanding. Instead, after running a cell, pause and ask: *What would happen if I doubled this value?* Then change it and run again. The handbook exercises exist to force that pause; treat them as part of the core workflow, not optional homework.

### Navigating the Repository

Start at [`book/SUMMARY.md`](../SUMMARY.md) — it is the table of contents and your progress tracker. Part I (sections 01–04) is fully written toward a complete path through math, PyTorch, neural networks, and transformers. Part II (section 05) covers diffusion models, reinforcement learning, and reading research papers — stubs today, expanded as you advance.

The root [`book/README.md`](../README.md) summarizes quick start commands and writing standards. Upstream inspiration comes from [Become Elite AI Researcher](https://github.com/vukrosic/become-elite-ai-researcher) by Vuk Rosić; this handbook is a personal adaptation for structured study, not a redistribution of the original notebooks.

### Cursor Rules and Skills for Authors

If you use Cursor to extend this handbook — or if an agent helps you draft chapters — two project files define quality expectations:

- **[`.cursor/rules/handbook-textbook.mdc`](../../.cursor/rules/handbook-textbook.mdc)** — audience assumptions, required 13-section structure for notebook chapters, callout types, notation rules, and anti-patterns (e.g., never summarize notebook cells without expanding them).
- **[`.cursor/skills/write-handbook-chapter/`](../../.cursor/skills/write-handbook-chapter/)** — workflow for transforming a lab notebook into a full chapter, including the chapter-to-notebook mapping and a gold-standard reference (`book/01-math/01-functions.md`).

Intro chapters like this one use a shorter template ([`book/_template/intro-chapter-template.md`](../_template/intro-chapter-template.md)). Full notebook chapters follow [`book/_template/chapter-template.md`](../_template/chapter-template.md) with sections from Introduction through Preview.

> 🧠 AI Insight
>
> Modern AI tooling can draft explanations quickly, but it defaults to shallow summaries. The handbook rules exist to force depth: intuition before notation, runnable code, explicit AI connections, and exercises at four difficulty tiers. When you prompt an agent to write or revise a chapter, cite the rule file and ask it to run validation — otherwise you get prose that *sounds* like a textbook but misses derivations and misconceptions sections.

### Validating Chapters

Before considering a chapter complete (yours or an agent's), run the validator:

```bash
just validate-book
just validate-book --chapter book/01-math/01-functions.md
```

The script (`scripts/validate_chapter.py`) checks word counts, required sections, Python code blocks, callout types, exercise tiers, interview questions, and AI keyword coverage. Intro chapters in `00-intro/` have lighter requirements (≥800 words, all four callouts) but still must pass validation.

Use validation as a **quality gate**, not a creativity limit. It catches missing sections and thin chapters; it does not replace reading the prose yourself.

> 🔬 Deep Dive
>
> The validator distinguishes intro chapters from notebook chapters via the path: anything under `book/00-intro/` skips the 13-section and 2,500-word requirements. Full chapters under `01-math`, `02-pytorch`, `03-neural-networks`, and `04-transformers` must include ≥5 Python blocks, ≥3 interview Q&A pairs, and references to AI concepts (gradients, transformers, loss functions, etc.). Running `just validate-book` without `--chapter` validates all handbook and intro files in those directories — useful before a large commit.

### How Long Should Each Session Be?

There is no fixed pace. A math chapter might take two evenings; a PyTorch indexing chapter might take one focused hour. What matters is **completion with understanding**, not speed. Block 60–90 minutes, finish one handbook section plus its lab cells, and stop while you can still explain the idea aloud. If you cannot explain it, you are not done — reread Intuition, rerun the notebook, try an Easy exercise.

### Working With Others

This handbook is personal by default, but the two-layer model scales to study groups: one person reads aloud from the handbook while others mutate the notebook. Teaching a concept to someone else — or to a rubber duck — is the fastest check that you actually understand it.

## Summary

- The handbook uses a **two-layer model**: `book/` for textbook depth, `app/` for runnable labs.
- Follow the **read → run → experiment → exercise → check off** workflow for every chapter.
- Authors and agents should follow **`.cursor/rules/handbook-textbook.mdc`** and **`write-handbook-chapter`** skill conventions.
- Run **`just validate-book`** (optionally with `--chapter`) before treating a chapter as finished.
- Track progress in **`book/SUMMARY.md`** and start with [Prerequisites](02-prerequisites.md) if you have not set up the environment yet.

## Further Reading

- [Prerequisites](02-prerequisites.md) — environment setup and assumed background
- [Learning Path](03-learning-path.md) — full 0→100 curriculum arc
- [`book/README.md`](../README.md) — quick start and repository structure
- [Become Elite AI Researcher](https://github.com/vukrosic/become-elite-ai-researcher) — upstream reference curriculum
