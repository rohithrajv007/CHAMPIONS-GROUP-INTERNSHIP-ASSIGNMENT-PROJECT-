# src/pipeline.py
import pandas as pd
import json
import sys
from src.preprocess import preprocess_dataframe
from src.scorer import CertificationScorer

def run_pipeline(input_csv, output_json):
    # Load CSV
    df = pd.read_csv(input_csv)

    # Preprocess text
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

    # Sort by descending score
    results.sort(key=lambda x: x["confidence_score"], reverse=True)

    # Save to JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Scoring complete — results saved to {output_json}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m src.pipeline <input_csv> <output_json>")
        sys.exit(1)
    
    run_pipeline(sys.argv[1], sys.argv[2])
