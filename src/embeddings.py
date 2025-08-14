# src/embeddings.py
from sentence_transformers import SentenceTransformer

_model = None

def get_embedding_model(model_name: str):
    """Load and cache the Sentence-BERT model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model

def encode_texts(texts: list[str], model_name: str):
    """Generate embeddings for a list of texts."""
    model = get_embedding_model(model_name)
    return model.encode(texts)
