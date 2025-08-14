# src/matcher.py
from rapidfuzz import fuzz
from src.config import CERTIFICATE_VARIANTS, FUZZY_THRESHOLD

def normalize_text(text: str) -> str:
    return text.lower().strip()

def check_certification(certifications: str) -> bool:
    """Check if the given certifications string contains the target AWS cert."""
    if not certifications:
        return False
    certified_text = normalize_text(certifications)
    for variant in CERTIFICATE_VARIANTS:
        # Exact substring match
        if variant in certified_text:
            return True
        # Fuzzy partial match
        if fuzz.partial_ratio(variant, certified_text) >= FUZZY_THRESHOLD:
            return True
    return False
