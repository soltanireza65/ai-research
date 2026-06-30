# Chapter Quality Checklist

## Before submitting

- [ ] All 13 section headings present (exact `## N. Title` format)
- [ ] ≥2,500 words (intro chapters ≥800)
- [ ] All four standard callout types used (`💡` `⚠️` `🧠` `🔬`)
- [ ] `📌 Preview` callout when ML terms appear before teaching chapter
- [ ] ≥5 Python code blocks
- [ ] Exercise tiers: Easy, Medium, Hard, Challenge
- [ ] ≥3 interview Q&A pairs (`**Q1:**` / `**A1:**`)
- [ ] Early math 01–04: Section 7 stays short; previews in callouts
- [ ] Chapter cheatsheet + `## Review` footer links
- [ ] No "it is obvious"
- [ ] `just validate-book --chapter <path>` passes

## Preview discipline

| If you mention… | Before teaching chapter, use… |
|-----------------|------------------------------|
| ReLU, sigmoid | `📌 Preview` → Single Neuron |
| MSE, loss | `📌 Preview` → Gradients |
| softmax, cross-entropy | `📌 Preview` → Probability |
| backprop, `loss.backward()` | `📌 Preview` → Backpropagation |
| attention, Q/K/V | `📌 Preview` → Attention chapters |

Map: [Vocabulary Roadmap](../../book/00-intro/04-vocabulary-roadmap.md)

## Anti-patterns

| Bad | Good |
|-----|------|
| Sigmoid formula in Functions §7 | Preview table + link to Single Neuron |
| Reader feels stupid for unknown word | `📌 Preview — optional for now` |
| Section 7 longer than core math in ch. 1–4 | Short AI teaser + roadmap link |
