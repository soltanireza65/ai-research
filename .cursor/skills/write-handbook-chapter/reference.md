# Chapter Quality Checklist

## Before submitting

- [ ] All 13 section headings present (exact `## N. Title` format)
- [ ] ≥2,500 words (intro chapters ≥800)
- [ ] All four callout types used
- [ ] ≥5 Python code blocks
- [ ] Exercise tiers: Easy, Medium, Hard, Challenge
- [ ] ≥3 interview Q&A pairs (`**Q1:**` / `**A1:**`)
- [ ] AI Connection mentions ML, neural networks, optimization, transformers, or embeddings
- [ ] No "it is obvious"
- [ ] Every graph description explains axes
- [ ] Notation explained on first use
- [ ] `just validate-book --chapter <path>` passes

## Anti-patterns

| Bad | Good |
|-----|------|
| "This notebook covers derivatives." | Step-by-step derivation with intuition first |
| `f'(x) = 2x` without explanation | Define f, x, derivative as rate of change |
| Code block with no commentary | Explain what each line does |
| 3 bullet summary of notebook | Full worked examples + exercises |
| Skipping AI connection | Concrete link to backprop, attention, etc. |

## Section depth guide

| Section | Minimum content |
|---------|-----------------|
| Introduction | Why + where in AI + outcomes |
| Intuition | Analogy + ASCII diagram |
| Formal Definitions | Every symbol defined |
| Programming | Math ↔ Python mapping |
| Visualizations | 2+ plots with axis labels explained |
| Worked Examples | 3+ examples, easy → hard |
| AI Connection | 2+ concrete ML/DL references |
| Common Mistakes | 2+ misconceptions |
| Exercises | 3+ per tier |
| Mini Project | Runnable spec |
| Interview | 3+ detailed answers |
| Summary | Formulas + terminology lists |
| Preview | Bridge to next chapter |
