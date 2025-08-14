# src/config.py

# Target certification we are checking for (Scenario 1)
TARGET_CERT = "aws certified machine learning – specialty"

# Known alternate variants for fuzzy matching
CERTIFICATE_VARIANTS = [
    "aws certified machine learning – specialty",
    "aws certified ml – specialty",
    "aws machine learning specialty",
    "aws ml specialty",
]

# Fuzzy matching threshold
FUZZY_THRESHOLD = 90

# Embedding model name (Sentence-BERT)
MODEL_NAME = "all-MiniLM-L6-v2"

# Similarity weights
SKILLS_WEIGHT = 0.5
WORK_EXP_WEIGHT = 0.5

# Max confidence when there is no direct certification
MAX_NO_CERT_CONFIDENCE = 95.0
