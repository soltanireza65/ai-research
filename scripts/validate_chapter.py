"""Validate handbook chapters meet textbook quality standards."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "book"

REQUIRED_SECTIONS = [
    "## 1. Introduction",
    "## 2. Intuition",
    "## 3. Formal Definitions",
    "## 4. Programming Perspective",
    "## 5. Visualizations",
    "## 6. Worked Examples",
    "## 7. AI Connection",
    "## 8. Common Mistakes",
    "## 9. Exercises",
    "## 10. Mini Project",
    "## 11. Interview Questions",
    "## 12. Summary",
    "## 13. Preview",
]

INTRO_REQUIRED_HEADINGS = ["## Learning Objectives", "## Content", "## Summary"]

CALLOUTS = ["💡 Intuition", "⚠️ Common Mistake", "🧠 AI Insight", "🔬 Deep Dive"]

PREVIEW_CALLOUT = "📌 Preview"

EXERCISE_SUBSECTIONS = ["### Easy", "### Medium", "### Hard", "### Challenge"]

AI_KEYWORDS = [
    "machine learning",
    "neural network",
    "transformer",
    "optimization",
    "embedding",
    "gradient",
    "backprop",
    "attention",
    "pytorch",
    "loss",
]

PREVIEW_TERMS = [
    "sigmoid",
    "relu",
    "softmax",
    "cross-entropy",
    "cross entropy",
    "embedding",
    "attention",
    "backprop",
    "gelu",
    "autograd",
]

CHAPTERS_WITH_LABS = {
    "01-functions.md",
    "02-derivatives.md",
    "03-vectors.md",
    "04-gradients.md",
    "05-matrices.md",
    "06-probability.md",
    "01-creating-tensors.md",
    "02-matrix-multiplication.md",
    "03-transposing-tensors.md",
    "04-reshaping-tensors.md",
    "05-indexing-and-slicing.md",
    "06-concatenating-tensors.md",
    "07-special-tensors.md",
    "01-single-neuron.md",
    "02-building-a-layer.md",
    "03-backpropagation.md",
    "01-attention-mechanism.md",
    "02-self-attention.md",
    "03-multi-head-attention.md",
    "04-decoder-only-transformer.md",
}

INTRO_MIN_WORDS = 800
CHAPTER_MIN_WORDS = 2500
CHEATSHEET_MIN_WORDS = 80

EARLY_MATH_CHAPTERS = {
    "01-functions.md",
    "02-derivatives.md",
    "03-vectors.md",
    "04-gradients.md",
}

PYTORCH_CHAPTERS = {
    "01-creating-tensors.md",
    "02-matrix-multiplication.md",
    "03-transposing-tensors.md",
    "04-reshaping-tensors.md",
    "05-indexing-and-slicing.md",
    "06-concatenating-tensors.md",
    "07-special-tensors.md",
}


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def count_python_blocks(text: str) -> int:
    return len(re.findall(r"```python\n", text))


def count_interview_qa(text: str) -> int:
    return len(re.findall(r"^\*\*Q\d*:", text, re.MULTILINE))


def is_intro_chapter(path: Path) -> bool:
    return "00-intro" in path.parts


def is_stub_chapter(path: Path) -> bool:
    return "05-future" in path.parts


def is_cheatsheet(path: Path) -> bool:
    return path.name.endswith("-cheatsheet.md")


def is_early_math(path: Path) -> bool:
    return "01-math" in path.parts and path.name in EARLY_MATH_CHAPTERS


def is_pytorch_chapter(path: Path) -> bool:
    return "02-pytorch" in path.parts and path.name in PYTORCH_CHAPTERS


def cheatsheet_path_for(chapter: Path) -> Path:
    return chapter.with_name(chapter.stem + "-cheatsheet.md")


def section_word_count(text: str, heading: str, next_headings: list[str]) -> int:
    start = text.find(heading)
    if start < 0:
        return 0
    start += len(heading)
    end = len(text)
    for nh in next_headings:
        pos = text.find(nh, start)
        if pos >= 0:
            end = min(end, pos)
    return word_count(text[start:end])


def preview_terms_in_text(text: str) -> list[str]:
    lower = text.lower()
    return [t for t in PREVIEW_TERMS if t in lower]


def core_sections_text(text: str) -> str:
    """Sections 1–6 (before AI Connection)."""
    start = text.find("## 1. Introduction")
    end = text.find("## 7. AI Connection")
    if start < 0:
        return text
    if end < 0:
        return text[start:]
    return text[start:end]


def resolve_markdown_link(source: Path, target: str) -> Path | None:
    if target.startswith("http"):
        return None
    base = source.parent
    return (base / target).resolve()


def validate_cheatsheet(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    wc = word_count(text)
    if wc < CHEATSHEET_MIN_WORDS:
        errors.append(f"Cheatsheet word count {wc} < minimum {CHEATSHEET_MIN_WORDS}")

    if "Vocabulary Roadmap" not in text:
        errors.append("Cheatsheet missing Vocabulary Roadmap link")

    for match in re.finditer(r"\]\(([^)]+)\)", text):
        target = match.group(1)
        if "00-intro/04-vocabulary-roadmap" in target:
            resolved = resolve_markdown_link(path, target)
            if resolved and not resolved.exists():
                errors.append(f"Broken roadmap link: {target}")

    if "Stuck" not in text and "stuck" not in text.lower():
        errors.append('Cheatsheet missing "Stuck?" guidance line')

    return errors


def validate_stub(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    if "**Status:**" not in text and "Stub" not in text:
        errors.append("Stub chapter missing status banner")
    if "## Prerequisites" not in text:
        errors.append("Stub chapter missing ## Prerequisites")
    if "Learning Path" not in text:
        errors.append("Stub chapter missing link back to Learning Path")
    return errors


def validate(path: Path) -> list[str]:
    errors: list[str] = []
    if is_cheatsheet(path):
        return validate_cheatsheet(path)

    if is_stub_chapter(path):
        return validate_stub(path)

    text = path.read_text(encoding="utf-8")
    intro = is_intro_chapter(path)

    if intro:
        wc = word_count(text)
        if wc < INTRO_MIN_WORDS:
            errors.append(f"Intro word count {wc} < minimum {INTRO_MIN_WORDS}")
        for heading in INTRO_REQUIRED_HEADINGS:
            if heading not in text:
                errors.append(f"Intro missing section: {heading}")
    else:
        for section in REQUIRED_SECTIONS:
            if section not in text:
                errors.append(f"Missing section: {section}")

        for subsection in EXERCISE_SUBSECTIONS:
            if subsection not in text:
                errors.append(f"Missing exercise tier: {subsection}")

        wc = word_count(text)
        if wc < CHAPTER_MIN_WORDS:
            errors.append(f"Word count {wc} < minimum {CHAPTER_MIN_WORDS}")

        blocks = count_python_blocks(text)
        if blocks < 5:
            errors.append(f"Only {blocks} Python code blocks (need ≥ 5)")

        early_math = is_early_math(path)
        pytorch = is_pytorch_chapter(path)

        ai_min = 1 if early_math else 3
        ai_hits = sum(1 for kw in AI_KEYWORDS if kw.lower() in text.lower())
        if ai_hits < ai_min:
            errors.append(f"AI keyword hits {ai_hits} < minimum {ai_min}")

        qa_count = count_interview_qa(text)
        if qa_count < 3:
            errors.append(f"Only {qa_count} interview questions (need ≥ 3)")

        if "it is obvious" in text.lower() or "it's obvious" in text.lower():
            errors.append('Contains forbidden phrase "it is obvious"')

        cs_path = cheatsheet_path_for(path)
        if not cs_path.exists():
            errors.append(f"Missing cheatsheet: {cs_path.name}")

        if path.name in CHAPTERS_WITH_LABS and "## Lab" not in text:
            errors.append("Chapter with companion notebook missing ## Lab section")

        if "Readiness checks" not in text:
            errors.append('Missing "### Readiness checks" in §12 Summary')

        if "Suggested pacing" not in text:
            errors.append("Missing Suggested pacing block in §1 Introduction")

        core = core_sections_text(text)
        if early_math or pytorch:
            terms = preview_terms_in_text(core)
            if terms and PREVIEW_CALLOUT not in text:
                errors.append(
                    f"Chapter mentions {terms[:3]} in §1–§6 but missing '{PREVIEW_CALLOUT}' callout"
                )

        if early_math:
            sec3 = section_word_count(text, "## 3. Formal Definitions", ["## 4."])
            sec7 = section_word_count(text, "## 7. AI Connection", ["## 8."])
            if sec3 > 0 and sec7 > sec3 * 1.5:
                errors.append(
                    f"Section 7 ({sec7} words) much longer than Section 3 ({sec3} words); "
                    "reduce AI preview drift in early math"
                )

    callout_hits = sum(1 for c in CALLOUTS if c in text)
    if callout_hits < 4:
        errors.append(f"Only {callout_hits}/4 standard callout types used")

    return errors


def collect_chapters(chapter_arg: str | None, include_stubs: bool) -> list[Path]:
    if chapter_arg:
        path = Path(chapter_arg)
        if not path.is_absolute():
            path = ROOT / path
        return [path]

    chapters: list[Path] = []
    for pattern in ["01-math", "02-pytorch", "03-neural-networks", "04-transformers"]:
        chapters.extend(sorted((BOOK / pattern).glob("*.md")))
    chapters.extend(sorted((BOOK / "00-intro").glob("*.md")))
    if include_stubs:
        chapters.extend(sorted((BOOK / "05-future").glob("*.md")))

    cheatsheets: list[Path] = []
    for pattern in ["00-intro", "01-math", "02-pytorch", "03-neural-networks", "04-transformers"]:
        cheatsheets.extend(sorted((BOOK / pattern).glob("*-cheatsheet.md")))

    all_paths = chapters + cheatsheets
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in all_paths:
        if p in seen:
            continue
        seen.add(p)
        unique.append(p)
    return [
        p
        for p in unique
        if p.name != "README.md" and not p.name.startswith("_")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate handbook chapters")
    parser.add_argument("--chapter", help="Validate a single chapter path")
    parser.add_argument(
        "--include-stubs",
        action="store_true",
        help="Also validate 05-future stub chapters",
    )
    args = parser.parse_args()

    chapters = collect_chapters(args.chapter, args.include_stubs)
    if not chapters:
        print("No chapters found to validate.")
        return 1

    failed = False
    for path in chapters:
        if not path.exists():
            print(f"FAIL {path}: file not found")
            failed = True
            continue
        errors = validate(path)
        if errors:
            failed = True
            print(f"FAIL {path.relative_to(ROOT)}")
            for err in errors:
                print(f"  - {err}")
        else:
            print(f"OK   {path.relative_to(ROOT)}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
