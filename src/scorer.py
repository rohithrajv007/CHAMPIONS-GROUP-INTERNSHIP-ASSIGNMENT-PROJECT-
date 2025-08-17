import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from src.embeddings import encode_texts
from src.matcher import check_certification
from src.config import (
    SKILLS_WEIGHT,
    WORK_EXP_WEIGHT,
    MAX_NO_CERT_CONFIDENCE,
    MODEL_NAME
)
from src.config import RELEVANT_SKILLS


# Path where your reference embeddings are stored
ARTIFACTS_PATH = "artifacts/reference_embeddings.joblib"

# Whitelist of skills relevant to AWS ML Certification
RELEVANT_SKILLS = set([
    "python", "amazon sagemaker", "aws lambda", "feature engineering",
    "machine learning", "data science", "model deployment", "mlops",
    "amazon rekognition", "aws", "tensorflow", "pytorch",
    "keras", "scikit-learn", "aws cloudformation", "aws step functions",
    "data engineering", "cloud computing", "deep learning"
])

def filter_relevant_skills(skills_text):
    """
    Filter the skills text to only include relevant AWS ML skills.
    Prevents unrelated skills from reducing similarity score.
    """
    if not isinstance(skills_text, str):
        return ""
    skills_list = [s.strip().lower() for s in skills_text.split(",")]
    filtered = [skill for skill in skills_list if skill in RELEVANT_SKILLS]
    return ", ".join(filtered) if filtered else skills_text.lower()

class CertificationScorer:
    def __init__(self):
        """Load reference embeddings and model info from joblib."""
        data = joblib.load(ARTIFACTS_PATH)
        self.skills_ref = data["skills_ref"].reshape(1, -1)
        self.work_ref = data["work_ref"].reshape(1, -1)
        self.model_name = data.get("model_name", MODEL_NAME)

    def score_profile(self, skills_text, work_text, certifications_text):
        """
        Score a LinkedIn profile for AWS ML certification likelihood.

        Returns:
            (certified_status: bool, confidence_score: float)
        """
        # ---------- Scenario 1: Direct certification match ----------
        if check_certification(certifications_text):
            return True, 100.0

        # ---------- Scenario 2: Similarity scoring ----------
        # Filter skills so irrelevant items don't drag down similarity
        filtered_skills = filter_relevant_skills(skills_text)

        # Encode both skills and work experience
        skills_embedding = encode_texts([filtered_skills], self.model_name)
        work_embedding = encode_texts(
            [work_text.lower() if isinstance(work_text, str) else ""],
            self.model_name
        )

        # Compute cosine similarities with certified reference embeddings
        skills_sim = cosine_similarity(skills_embedding, self.skills_ref)[0][0]
        work_sim = cosine_similarity(work_embedding, self.work_ref)[0][0]

        # Weighted similarity score (0-100)
        raw_score = 100 * (SKILLS_WEIGHT * skills_sim + WORK_EXP_WEIGHT * work_sim)

        # Cap score if no certification found
        capped_score = min(raw_score, MAX_NO_CERT_CONFIDENCE)

        return False, round(float(capped_score), 2)
