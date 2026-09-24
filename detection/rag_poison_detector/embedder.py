from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

class DocumentEmbedder:
    def __init__(self, device = 'cpu'):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)   

    def embed(self, documents):
        return self.model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)