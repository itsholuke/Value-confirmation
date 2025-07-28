# instagram_preprocess.py
"""
Minimal script to transform Instagram post data from the raw format
(ig_posts_raw_mini.csv) into the tokenised format expected in
ig_posts_transformed_mini.csv.

Input columns  : shortcode, caption, … (other columns ignored)
Output columns : ID, Context, Sentence ID, Statement

Transformation rules
--------------------
* ``shortcode``   -> ``ID``
* ``caption``     -> ``Context`` (left intact)
* Each sentence from ``Context`` becomes its own row in ``Statement``.
* Hashtags (e.g. #summer) are preserved as standalone sentences.
* Rows that would contain only punctuation are dropped.
* ``Sentence ID`` counts sentences sequentially (starting at 1) per post.

The script is deliberately dependency‑light (only ``pandas`` and the Python
standard library) so it can run on vanilla Google Colab without extra setup.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Sentence tokenizer
# ---------------------------------------------------------------------------

_HASH_RE = re.compile(r"#\w+")
_PUNCT_RE = re.compile(r"[.!?]+")  # sentence‑ending punctuation
_WORD_RE = re.compile(r"\w")       # at least one word‑character

# A quick, reasonably robust splitter that
#   * Treats hashtags as atomic sentences.
#   * Splits on ., !, ? followed by whitespace.
#   * Keeps emoji and other unicode because they can be informative.

def split_sentences(text: str) -> list[str]:
    """Return a list of sentences extracted from *text*.

    Hashtags are emitted as separate sentences; punctuation‑only fragments are
    discarded.
    """
    if not text:
        return []

    text = text.replace("\n", " ").strip()

    # Tokenise into hashtag / punctuation / other chunks
    tokens = re.findall(r"#\w+|[.!?]+|[^#.!?]+", text)

    sentences: list[str] = []
    buffer = ""

    for tok in tokens:
        if tok.startswith("#"):
            # Flush any buffered sentence before the hashtag
            if buffer.strip():
                sentences.extend(
                    [s.strip() for s in re.split(r"(?<=[.!?])\s+", buffer) if _WORD_RE.search(s)]
                )
                buffer = ""
            sentences.append(tok.strip())
        elif _PUNCT_RE.fullmatch(tok):
            buffer += tok  # keep punctuation attached to the prior chunk
        else:
            buffer += tok

    # Trailing buffer
    if buffer.strip():
        sentences.extend(
            [s.strip() for s in re.split(r"(?<=[.!?])\s+", buffer) if _WORD_RE.search(s)]
        )

    # Final sanity prune – remove stray punct‑only bits just in case
    sentences = [s for s in sentences if _WORD_RE.search(s)]
    return sentences

# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def transform_raw_csv(raw_csv: str | Path, out_csv: str | Path = "ig_posts_transformed_mini.csv") -> None:
    """Read *raw_csv*, transform, and write *out_csv*."""
    df_raw = pd.read_csv(raw_csv)

    # Rename required columns for consistency
    df_raw = df_raw.rename(columns={"shortcode": "ID", "caption": "Context"})

    records: list[dict[str, str | int]] = []

    for _, row in df_raw.iterrows():
        sentences = split_sentences(str(row["Context"]))
        for idx, sent in enumerate(sentences, start=1):
            records.append(
                {
                    "ID": row["ID"],
                    "Context": row["Context"],
                    "Sentence ID": idx,
                    "Statement": sent,
                }
            )

    df_out = pd.DataFrame(records, columns=["ID", "Context", "Sentence ID", "Statement"])
    df_out.to_csv(out_csv, index=False)
    print(f"✅ Wrote {len(df_out):,} sentence rows to {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python instagram_preprocess.py <ig_posts_raw_mini.csv> [output.csv]", file=sys.stderr)
        sys.exit(1)

    raw_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "ig_posts_transformed_mini.csv"
    transform_raw_csv(raw_path, out_path)
