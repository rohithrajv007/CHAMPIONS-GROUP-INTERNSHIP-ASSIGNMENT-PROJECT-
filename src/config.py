# Target certification (Scenario 1 exact/fuzzy matching)
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

# ✅ NEW: Whitelist of relevant skills for AWS ML-certified profiles
RELEVANT_SKILLS = set([
    "python", "amazon sagemaker", "aws lambda", "feature engineering",
    "machine learning", "data science", "model deployment", "mlops",
    "amazon rekognition", "aws", "tensorflow", "pytorch",
    "keras", "scikit-learn", "aws cloudformation", "aws step functions",
    "data engineering", "cloud computing", "deep learning"
])
