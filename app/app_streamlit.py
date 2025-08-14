import sys
import os

# Ensure project root is in sys.path so "src" can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import json
from src.preprocess import preprocess_dataframe
from src.scorer import CertificationScorer

st.set_page_config(page_title="LinkedIn AWS Certification Predictor", layout="wide")

st.title("🚀 AWS Certification Likelihood Predictor")
st.write("Upload a CSV file of LinkedIn user profiles and get a classification report in JSON format, sorted by confidence score.")

# File uploader
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        
        # Preprocess
        df = preprocess_dataframe(df)

        # Init scorer
        scorer = CertificationScorer()

        results = []
        for _, row in df.iterrows():
            certified_status, confidence_score = scorer.score_profile(
                row["skills"], row["work_experience"], row["certifications"]
            )
            results.append({
                "name": row["name"],
                "certified_status": certified_status,
                "confidence_score": round(float(confidence_score), 2)
            })

        # Sort by score desc
        results = sorted(results, key=lambda x: x["confidence_score"], reverse=True)

        # Display results table
        st.subheader("📊 Classification Results")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)

        # Save JSON to bytes for download
        json_bytes = json.dumps(results, indent=2).encode("utf-8")

        st.download_button(
            label="💾 Download Results as JSON",
            data=json_bytes,
            file_name="classification_results.json",
            mime="application/json"
        )

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("Please upload a CSV file to get started.")
