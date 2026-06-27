# AI Research Handbook

A personal learning handbook based on [Become Elite AI Researcher](https://github.com/vukrosic/become-elite-ai-researcher) by Vuk Rosić. This is an adaptation for structured study — not a redistribution of the original notebooks.

## Quick start

```bash
uv sync
just lab
```

Open [`SUMMARY.md`](SUMMARY.md) and start with [How to Use This Handbook](00-intro/01-how-to-use-this-handbook.md).

## Study workflow

For each chapter:

1. **Read** the chapter markdown in `book/`
2. **Run** the companion notebook in `app/` (`just lab`)
3. **Experiment** — change values, break things, observe
4. **Complete** exercises at the end of the chapter
5. **Check off** the chapter in [`SUMMARY.md`](SUMMARY.md)

## Structure

| Layer | Location | Role |
|-------|----------|------|
| Handbook | `book/<module>/` | Full textbook chapters (intuition, derivations, exercises) |
| Labs | `app/<module>/` | Runnable notebooks |
| Utilities | `app/utils/` | Shared plotting helpers |
| Assets | `book/assets/` | Saved figures from labs |

## Writing standards

Chapters follow the 13-section textbook template in [`_template/chapter-template.md`](_template/chapter-template.md). Cursor agents use:

- [`.cursor/rules/handbook-textbook.mdc`](../.cursor/rules/handbook-textbook.mdc)
- [`.cursor/skills/write-handbook-chapter/`](../.cursor/skills/write-handbook-chapter/)

Validate chapters with:

```bash
just validate-book
just validate-book --chapter book/01-math/01-functions.md
```

## Attribution

> Based on [Become Elite AI Researcher](https://github.com/vukrosic/become-elite-ai-researcher) by Vuk Rosić. This handbook is a personal adaptation for learning, not a redistribution of the original notebooks.
