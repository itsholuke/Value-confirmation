import streamlit as st
import pandas as pd
import re
import json
from io import StringIO

st.set_page_config(page_title="Marketing Keyword Classifier", page_icon="🔍", layout="centered")

st.title("🔍 Marketing Keyword Classifier")
st.markdown(
    """Upload any CSV containing free‑text statements (e.g. product descriptions, email subject lines, ad copy) and quickly flag phrases that create *urgency* or *exclusivity*.\n\n---"""
)

# ---------------------
# 1) Upload data
# ---------------------
uploaded_file = st.file_uploader("📄 Upload CSV", type=["csv"], accept_multiple_files=False)

df: pd.DataFrame | None = None
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset loaded! Preview below ⬇")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"❌ Could not read CSV: {e}")

# ---------------------
# 2) Configure / extend dictionaries
# ---------------------
DEFAULT_DICTIONARIES = {
    "urgency_marketing": [
        "limited", "limited time", "limited run", "limited edition", "order now",
        "last chance", "hurry", "while supplies last", "before they're gone",
        "selling out", "selling fast", "act now", "don't wait", "today only",
        "expires soon", "final hours", "almost gone"
    ],
    "exclusive_marketing": [
        "exclusive", "exclusively", "exclusive offer", "exclusive deal",
        "members only", "vip", "special access", "invitation only",
        "premium", "privileged", "limited access", "select customers",
        "insider", "private sale", "early access"
    ],
}

st.markdown("### 🛠 Edit keyword dictionaries (JSON)")

with st.expander("Show/Hide dictionaries editor"):
    json_input = st.text_area(
        "Each top‑level key is a category; provide a JSON object mapping to *lists* of keywords.",
        value=json.dumps(DEFAULT_DICTIONARIES, indent=4),
        height=300,
    )

# Parse custom dictionaries (fall back to default on error)
try:
    dictionaries = json.loads(json_input)
    if not isinstance(dictionaries, dict):
        raise ValueError("Top‑level JSON must be an object mapping category ➜ keywords list.")
    dictionaries = {
        cat: set(terms) if isinstance(terms, (list, set)) else set()
        for cat, terms in dictionaries.items()
    }
except Exception as e:
    st.warning(f"⚠ Invalid JSON provided – using default dictionaries. ({e})")
    dictionaries = {k: set(v) for k, v in DEFAULT_DICTIONARIES.items()}

# ---------------------
# 3) Build regex patterns
# ---------------------
patterns = {
    cat: re.compile(r"\\b(?:" + "|".join(map(re.escape, terms)) + r")\\b", flags=re.I)
    for cat, terms in dictionaries.items()
}

# ---------------------
# 4) Classification helper
# ---------------------
@st.cache_data(show_spinner=False)
def classify_series(series: pd.Series) -> pd.Series:
    def classify(text: str | float):
        text = text if isinstance(text, str) else ""
        cats = [cat for cat, pat in patterns.items() if pat.search(text)]
        return ",".join(cats) if cats else None

    return series.apply(classify)

# ---------------------
# 5) Choose column + run
# ---------------------
if df is not None:
    with st.form(key="run_form"):
        text_col = st.selectbox("Select the column containing text to analyze", options=df.columns, index=0)
        submitted = st.form_submit_button("🚀 Run classification")

    if submitted:
        with st.spinner("Classifying…"):
            df_result = df.copy()
            df_result["categories"] = classify_series(df_result[text_col])
        st.success("Done!")
        st.dataframe(df_result)

        # Offer download
        csv_bytes = df_result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download results as CSV",
            data=csv_bytes,
            file_name="classified_data.csv",
            mime="text/csv",
        )

# ---------------------
# Footer
# ---------------------
st.markdown("---")
st.caption("Built with ❤ using Streamlit.")
