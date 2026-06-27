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

CALLOUTS = ["💡 Intuition", "⚠️ Common Mistake", "🧠 AI Insight", "🔬 Deep Dive"]

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

INTRO_MIN_WORDS = 800
CHAPTER_MIN_WORDS = 2500


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def count_python_blocks(text: str) -> int:
    return len(re.findall(r"```python\n", text))


def count_interview_qa(text: str) -> int:
    return len(re.findall(r"^\*\*Q\d*:", text, re.MULTILINE))


def is_intro_chapter(path: Path) -> bool:
    return "00-intro" in path.parts


def validate(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    intro = is_intro_chapter(path)

    if not intro:
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

        ai_hits = sum(1 for kw in AI_KEYWORDS if kw.lower() in text.lower())
        if ai_hits < 3:
            errors.append(f"AI Connection too thin ({ai_hits} keyword hits, need ≥ 3)")

        qa_count = count_interview_qa(text)
        if qa_count < 3:
            errors.append(f"Only {qa_count} interview questions (need ≥ 3)")

        if "it is obvious" in text.lower() or "it's obvious" in text.lower():
            errors.append('Contains forbidden phrase "it is obvious"')
    else:
        wc = word_count(text)
        if wc < INTRO_MIN_WORDS:
            errors.append(f"Intro word count {wc} < minimum {INTRO_MIN_WORDS}")

    callout_hits = sum(1 for c in CALLOUTS if c in text)
    if callout_hits < 4:
        errors.append(f"Only {callout_hits}/4 callout types used")

    return errors


def collect_chapters(chapter_arg: str | None) -> list[Path]:
    if chapter_arg:
        path = Path(chapter_arg)
        if not path.is_absolute():
            path = ROOT / path
        return [path]
    chapters: list[Path] = []
    for pattern in ["01-math", "02-pytorch", "03-neural-networks", "04-transformers"]:
        chapters.extend(sorted((BOOK / pattern).glob("*.md")))
    chapters.extend(sorted((BOOK / "00-intro").glob("*.md")))
    return [p for p in chapters if p.name != "README.md"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate handbook chapters")
    parser.add_argument("--chapter", help="Validate a single chapter path")
    args = parser.parse_args()

    chapters = collect_chapters(args.chapter)
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
