from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from medical_assistant.config.personas import PERSONAS


def get_retriever(store: Chroma, persona: str) -> VectorStoreRetriever:
    k = PERSONAS[persona].retriever_k
    return store.as_retriever(search_kwargs={"k": k})
