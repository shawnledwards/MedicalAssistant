from pathlib import Path

from langchain_community.vectorstores import Chroma

from medical_assistant.config.personas import PERSONAS
from medical_assistant.config.settings import Settings
from medical_assistant.core.embeddings import get_embeddings
from medical_assistant.rag.document_loader import load_and_chunk

_stores: dict[str, Chroma] = {}


def get_vector_store(persona: str, settings: Settings) -> Chroma:
    if persona in _stores:
        return _stores[persona]

    config = PERSONAS[persona]
    embeddings = get_embeddings(settings)
    persist_dir = Path(settings.vector_db_path) / persona

    if persist_dir.exists() and any(persist_dir.iterdir()):
        store = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
        )
    else:
        persist_dir.mkdir(parents=True, exist_ok=True)
        chunks = load_and_chunk(
            pdf_path=settings.pdf_path,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(persist_dir),
        )

    _stores[persona] = store
    return store


def stores_ready() -> bool:
    return len(_stores) > 0
