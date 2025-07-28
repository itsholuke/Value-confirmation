#!/usr/bin/env python3
"""
Streamlit interface for Instagram post-preprocessing.
Run with:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import re
from typing import List

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Lightweight sentence tokenizer (shared with instagram_preprocess.py)
# ---------------------------------------------------------------------------
_HASH_RE = re.compile(r"#\w+")
_PUNCT_RE = re.compile(r"[.!?]+")
_WORD_RE = re.compile(r"\w")


def split_sentences(text: str) -> List[str]:
    """Return a list of sentences/hashtags from *text*."""
    if not text:
        return []
    text = text.replace("\n", " ").strip()
    tokens = re.findall(r"#\w+|[.!?]+|[^#.!?]+", text)
    sentences: List[str] = []
    buf = ""
    for tok in tokens:
        if tok.startswith("#"):
            if buf.strip():
                sentences.extend([
                    s.strip() for s in re.split(r"(?<=[.!?])\s+", buf) if _WORD_RE.search(s)
                ])
                buf = ""
            sentences.append(tok.strip())
        elif _PUNCT_RE.fullmatch(tok):
            buf += tok
        else:
            buf += tok
    if buf.strip():
        sentences.extend([
            s.strip() for s in re.split(r"(?<=[.!?])\s+", buf) if _WORD_RE.search(s)
        ])
    return [s for s in sentences if _WORD_RE.search(s)]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def transform_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Tokenise captions from the raw Instagram export (shortcode/caption)."""
    df_raw = df_raw.rename(columns={"shortcode": "ID", "caption": "Context"})
    rows = []
    for _, r in df_raw.iterrows():
        sents = split_sentences(str(r["Context"]))
        for idx, s in enumerate(sents, 1):
            rows.append({
                "ID": r["ID"],
                "Context": r["Context"],
                "Sentence ID": idx,
                "Statement": s,
            })
    return pd.DataFrame(rows)


def add_context_cols(df: pd.DataFrame, context_cut: str) -> pd.DataFrame:
    """Add a *Context* column based on the chosen cut (whole vs rolling)."""
    if context_cut == "whole":
        return df  # already whole

    # Rolling window: concatenate all previous statements in the same post
    contexts: List[str] = []
    for _, group in df.groupby("ID", sort=False):  # preserve original order
        current: List[str] = []
        for stmt in group["Statement"].tolist():
            current.append(stmt)
            contexts.append(" ".join(current))

    df_out = df.copy()
    df_out["Context"] = contexts
    return df_out

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="IG Preprocess", page_icon="✨", layout="wide")
st.title("✨ Instagram Post Preprocessing Demo")

with st.sidebar:
    st.header("⚙️ Options")
    st.markdown(
        "Upload either **ig_posts_raw_mini.csv** (with `shortcode`, `caption`) or an already tokenised file (`ID`, `Context`, `Sentence ID`, `Statement`)."
    )
    data_type = st.radio(
        "Input type", ["Raw Instagram export", "Tokenised (ID/Statement)"]
    )
    statement_cut = st.selectbox("Statement cut", ["sentence", "post"], index=0)
    context_cut = st.selectbox("Context cut", ["whole", "rolling"], index=0)

uploaded = st.file_uploader("📄 Choose a CSV file", type=["csv"])

if uploaded is not None:
    df_in = pd.read_csv(uploaded)

    if data_type == "Raw Instagram export":
        df_proc = transform_raw(df_in)
    else:
        # Validate expected columns exist
        expected = {"ID", "Context", "Sentence ID", "Statement"}
        missing = expected - set(df_in.columns)
        if missing:
            st.error(f"Uploaded file is missing columns: {', '.join(missing)}")
            st.stop()
        df_proc = df_in[list(expected)].copy()

    # Adjust statement cut (post-level aggregates all sentences per post)
    if statement_cut == "post":
        df_proc = (
            df_proc.groupby("ID", sort=False)
            .agg({"Context": "first", "Statement": " ".join})
            .reset_index()
        )
        df_proc["Sentence ID"] = 1
        df_proc = df_proc[["ID", "Context", "Sentence ID", "Statement"]]

    # Context selection (whole vs rolling)
    df_proc = add_context_cols(df_proc, context_cut)

    st.success(f"Processed {len(df_proc):,} rows ✅")
    st.dataframe(df_proc.head(50))

    csv_bytes = df_proc.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download CSV", csv_bytes, file_name="ig_posts_processed.csv", mime="text/csv"
    )
else:
    st.info("Upload a CSV file to begin.")