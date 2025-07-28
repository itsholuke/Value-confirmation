"""Streamlit interface for Instagram post pre‑processing.
Run with:  streamlit run streamlit_app.py"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Lightweight sentence tokenizer (same logic as in instagram_preprocess.py)
# ---------------------------------------------------------------------------
_HASH_RE = re.compile(r"#\w+")
_PUNCT_RE = re.compile(r"[.!?]+")
_WORD_RE = re.compile(r"\w")


def split_sentences(text: str) -> List[str]:
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
    """Tokenise captions from raw Instagram export."""
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
    """Add a Context column per selected cut (rolling vs whole)."""
    if "Context" not in df.columns:
        raise ValueError("Input DataFrame needs a Context column")
    if context_cut == "whole":
        # Already whole – ensure single copy of caption per post
        return df
    # rolling window (all previous statements in the same post)
    contexts = []
    for (pid), group in df.groupby("ID"):
        current = []
        for stmt in group["Statement"]:
            current.append(stmt)
            contexts.append(" ".join(current))
    df = df.copy()
    df["Context"] = contexts
    return df

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="IG Pre‑process", page_icon="🪄", layout="wide")
st.title("🪄 Instagram Post Pre‑processing Demo")

with st.sidebar:
    st.header("⚙️ Options")
    st.markdown("Upload either **ig_posts_raw_mini.csv** (with *shortcode/caption*) or an already tokenised file (*ID*, *Context*, *Sentence ID*, *Statement*).")
    data_type = st.radio("Input type", ["Raw Instagram export", "Tokenised (ID/Statement)"])
    statement_cut = st.selectbox("Statement cut", ["sentence", "post"], index=0)
    context_cut = st.selectbox("Context cut", ["whole", "rolling"], index=0)

uploaded = st.file_uploader("📄 Choose a CSV file", type=["csv"])

if uploaded is not None:
    df_in = pd.read_csv(uploaded)

    if data_type == "Raw Instagram export":
        df_proc = transform_raw(df_in)
    else:
        # Assume already tokenised; rename to expected columns if necessary
        df_proc = df_in.rename(columns=str.title)

    # Adjust statement cut (if post-level requested)
    if statement_cut == "post":
        # Aggregate statements per post/ID
        agg = (
            df_proc.groupby("ID")
            .agg({"Context": "first", "Statement": " ".join})
            .reset_index()
        )
        agg["Sentence ID"] = 1
        df_proc = agg[["ID", "Context", "Sentence ID", "Statement"]]

    # Context selection
    df_proc = add_context_cols(df_proc, context_cut)

    st.success(f"Processed {len(df_proc):,} rows ✅")
    st.dataframe(df_proc.head(50))

    # Download button
    csv_bytes = df_proc.to_csv(index=False).encode()
    st.download_button("⬇️ Download CSV", csv_bytes, file_name="ig_posts_processed.csv", mime="text/csv")

else:
    st.info("Upload a CSV file to begin.")
