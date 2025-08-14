import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from src.embeddings import encode_texts
from src.matcher import check_certification
from src.config import SKILLS_WEIGHT, WORK_EXP_WEIGHT, MAX_NO_CERT_CONFIDENCE, MODEL_NAME

ARTIFACTS_PATH = "artifacts/reference_embeddings.joblib"

class CertificationScorer:
    def __init__(self):
        # Load reference embeddings and model info
        data = joblib.load(ARTIFACTS_PATH)
        self.skills_ref = data["skills_ref"].reshape(1, -1)
        self.work_ref = data["work_ref"].reshape(1, -1)
        self.model_name = data.get("model_name", MODEL_NAME)

    def score_profile(self, skills_text, work_text, certifications_text):
        # Scenario 1: Direct certification match
        if check_certification(certifications_text):
            return True, 100.0
        
        # Scenario 2: No certification, calculate similarity scores
        skills_embedding = encode_texts([skills_text], self.model_name)
        work_embedding = encode_texts([work_text], self.model_name)

        skills_sim = cosine_similarity(skills_embedding, self.skills_ref)[0][0]
        work_sim = cosine_similarity(work_embedding, self.work_ref)[0][0]

        # Weighted similarity score (scale to 0-100)
        raw_score = 100 * (SKILLS_WEIGHT * skills_sim + WORK_EXP_WEIGHT * work_sim)

        # Cap the score to max confidence for no certification
        capped_score = min(raw_score, MAX_NO_CERT_CONFIDENCE)
        
        return False, capped_score
