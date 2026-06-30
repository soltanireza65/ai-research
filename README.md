# AI Research

Personal learning repo for becoming an AI engineer and eventually an AI researcher. The core artifact is the textbook-style handbook in [`book/`](book/), with runnable companion labs in [`app/`](app/).

The handbook is based on [Become Elite AI Researcher](https://github.com/vukrosic/become-elite-ai-researcher) by Vuk Rosić, expanded into original long-form chapters focused on intuition, math, visualization, code, and AI applications.

## Start Here

1. Open [`book/SUMMARY.md`](book/SUMMARY.md).
2. Read [`book/00-intro/01-how-to-use-this-handbook.md`](book/00-intro/01-how-to-use-this-handbook.md).
3. If unfamiliar terms like ReLU or sigmoid cause anxiety, read [`book/00-intro/04-vocabulary-roadmap.md`](book/00-intro/04-vocabulary-roadmap.md) before Chapter 1.
4. Run the companion notebooks with:

```bash
uv sync
just lab
```

## Structure

| Path | Purpose |
|------|---------|
| [`book/`](book/) | Textbook chapters and study workflow |
| [`app/`](app/) | Runnable Jupyter notebooks |
| [`app/utils/`](app/utils/) | Shared helpers for labs and plots |
| [`scripts/`](scripts/) | Notebook generation and chapter validation |
| [`.cursor/rules/`](.cursor/rules/) | Cursor rules that enforce textbook-quality chapter writing |
| [`.cursor/skills/`](.cursor/skills/) | Project skill for writing handbook chapters |

## Commands

```bash
just lab             # Open Jupyter Lab on app/
just notebooks       # Regenerate companion lab notebooks
just validate-book   # Validate handbook chapter structure and depth
just check           # Lint, typecheck, and test
```

## Writing Standard

Every notebook-sourced chapter is written as a real textbook chapter, not a summary. Chapters use the 13-section structure in [`book/_template/chapter-template.md`](book/_template/chapter-template.md) and are validated by [`scripts/validate_chapter.py`](scripts/validate_chapter.py).

## Attribution

This is a personal adaptation for learning, not a redistribution of the original course notebooks.
