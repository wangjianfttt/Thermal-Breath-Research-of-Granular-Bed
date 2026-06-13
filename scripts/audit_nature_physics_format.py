#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEX = ROOT / "thermal_ratcheting_nature_physics.tex"
OUT = ROOT / "nature_physics_format_audit.csv"


def strip_latex(text: str) -> str:
    text = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", " ", text, flags=re.S)
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.S)
    text = re.sub(r"\$.*?\$", " ", text, flags=re.S)
    text = re.sub(r"\\caption\{.*?\}", " ", text, flags=re.S)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", text)
    text = re.sub(r"[{}_^&%#~]", " ", text)
    text = text.replace("--", " ")
    return text


def word_count(text: str) -> int:
    stripped = strip_latex(text)
    words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", stripped)
    return len(words)


def main() -> None:
    text = TEX.read_text()
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.S).group(1)
    main_text = re.search(r"\\section\*\{Main\}(.*?)\\section\*\{Methods summary\}", text, re.S).group(1)
    methods = re.search(r"\\section\*\{Methods summary\}(.*?)\\section\*\{Data availability\}", text, re.S).group(1)
    figures = re.findall(r"\\begin\{figure\}", text)
    captions = re.findall(r"\\caption\{(.*?)\\label", text, re.S)
    rows = [
        ("abstract", word_count(abstract), "limit 200"),
        ("main_text_excluding_figures", word_count(main_text), "limit 3000"),
        ("methods_summary", word_count(methods), "limit 3000 online Methods"),
        ("figure_count", len(figures), "limit 6 main display items"),
        ("caption_words_total", sum(word_count(c) for c in captions), "each caption <350 words"),
    ]
    output_lines = [f"{key},{value},{note}" for key, value, note in rows]
    output_lines.append("caption_words_each," + ";".join(str(word_count(c)) for c in captions) + ",limit 350 each")
    OUT.write_text("\n".join(output_lines) + "\n")
    for key, value, note in rows:
        print(f"{key},{value},{note}")
    print("caption_words_each," + ";".join(str(word_count(c)) for c in captions) + ",limit 350 each")


if __name__ == "__main__":
    main()
