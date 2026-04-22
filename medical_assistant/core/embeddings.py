from typing import List

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from medical_assistant.config.settings import Settings

_embeddings_instance: "DirectEmbeddings | None" = None


class DirectEmbeddings(Embeddings):
    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode([text], convert_to_numpy=True)[0].tolist()


def get_embeddings(settings: Settings) -> DirectEmbeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = DirectEmbeddings(settings.embedding_model)
    return _embeddings_instance
