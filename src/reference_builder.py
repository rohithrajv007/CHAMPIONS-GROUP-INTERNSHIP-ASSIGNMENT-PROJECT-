# src/reference_builder.py
import os
import pandas as pd
import joblib
import numpy as np
from src.preprocess import preprocess_dataframe
from src.matcher import check_certification
from src.embeddings import encode_texts
from src.config import MODEL_NAME

DATA_PATH = "data/synthetic_profiles.csv"
ARTIFACTS_DIR = "artifacts"

def build_reference_embeddings():
    """Create and save average reference embeddings from certified profiles."""
    df = pd.read_csv(DATA_PATH)
    df = preprocess_dataframe(df)

    # Filter only certified profiles
    certified_df = df[df["certifications"].apply(check_certification)]

    if certified_df.empty:
        raise ValueError("No certified profiles found to build reference embeddings.")

    # Encode skills & work_experience separately
    skills_embeddings = encode_texts(certified_df["skills"].tolist(), MODEL_NAME)
    work_embeddings = encode_texts(certified_df["work_experience"].tolist(), MODEL_NAME)

    # Compute reference (average) embeddings
    skills_ref = np.mean(skills_embeddings, axis=0)
    work_ref = np.mean(work_embeddings, axis=0)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(
        {"skills_ref": skills_ref, "work_ref": work_ref, "model_name": MODEL_NAME},
        os.path.join(ARTIFACTS_DIR, "reference_embeddings.joblib")
    )
    print(f"✅ Reference embeddings saved to {ARTIFACTS_DIR}/reference_embeddings.joblib")

if __name__ == "__main__":
    build_reference_embeddings()
