# src/preprocess.py
import re
import pandas as pd

def clean_text(text: str) -> str:
    """Lowercase, remove punctuation, normalize whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text)      # collapse whitespace
    return text.strip()

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean skills, work_experience, and certifications columns."""
    for col in ["skills", "work_experience", "certifications"]:
        if col in df.columns:
            df[col] = df[col].fillna("").apply(clean_text)
    return df
