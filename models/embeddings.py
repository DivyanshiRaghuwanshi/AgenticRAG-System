from langchain_huggingface import HuggingFaceEmbeddings
from config.config import EMBEDDING_MODEL_NAME


def get_embedding_model():
    # all-MiniLM-L6-v2: free, local, CPU-friendly, ~23MB
    # normalize_embeddings=True allows cosine similarity to work correctly with FAISS
    try:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model: {e}")
