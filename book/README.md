# AI Research Handbook

A personal learning handbook based on [Become Elite AI Researcher](https://github.com/vukrosic/become-elite-ai-researcher) by Vuk Rosić. This is an adaptation for structured study — not a redistribution of the original notebooks.

## Quick start

```bash
uv sync
just lab
```

Open [`SUMMARY.md`](SUMMARY.md) and start with [How to Use This Handbook](00-intro/01-how-to-use-this-handbook.md). If AI jargon in early chapters worries you, read [Vocabulary Roadmap](00-intro/04-vocabulary-roadmap.md) first.

## Study workflow

For each chapter:

1. **Read** the chapter markdown in `book/`
2. **Run** the companion notebook in `app/` (`just lab`)
3. **Experiment** — change values, break things, observe
4. **Complete** Easy exercises first (skip Challenge previews if needed)
5. **Review** the chapter **cheatsheet** for retention
6. **Check** readiness items in §12 Summary before moving on
7. **Check off** the chapter in [`SUMMARY.md`](SUMMARY.md)

If jargon appears too early, check [Vocabulary Roadmap](00-intro/04-vocabulary-roadmap.md) or `📌 Preview` boxes.

**Weak math?** Start with [Math Basics](00-intro/05-math-basics.md) before Functions.

**Module review:** Before starting Module 03, skim all Module 02 cheatsheets.

§11 Interview Questions are **optional until Module 01 is complete**.

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

**Module-aware rules:** early math (`01`–`04`) and PyTorch chapters require `📌 Preview` callouts when ML jargon appears in §1–§6. Cheatsheets are validated separately (min length, roadmap links, Stuck? line). Stubs in `05-future/` use `--include-stubs`. See [`scripts/validate_chapter.py`](../scripts/validate_chapter.py).

## Attribution

> Based on [Become Elite AI Researcher](https://github.com/vukrosic/become-elite-ai-researcher) by Vuk Rosić. This handbook is a personal adaptation for learning, not a redistribution of the original notebooks.
