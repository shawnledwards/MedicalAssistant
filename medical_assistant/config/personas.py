from dataclasses import dataclass


@dataclass(frozen=True)
class PersonaConfig:
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    chunk_size: int
    chunk_overlap: int
    retriever_k: int


# Parameter rationale:
# scientist  — exhaustive context (k=10, large chunks), deterministic (temp=0.1), long answers
# doctor     — moderate context (k=5), warmer language (temp=0.35), mid-length answers
# faq        — minimal context (k=3), focused (low top_p/top_k), brief answers
PERSONAS: dict[str, PersonaConfig] = {
    "scientist": PersonaConfig(
        max_tokens=1536,
        temperature=0.1,
        top_p=0.95,
        top_k=50,
        chunk_size=1024,
        chunk_overlap=100,
        retriever_k=10,
    ),
    "doctor": PersonaConfig(
        max_tokens=768,
        temperature=0.35,
        top_p=0.90,
        top_k=40,
        chunk_size=512,
        chunk_overlap=50,
        retriever_k=5,
    ),
    "faq": PersonaConfig(
        max_tokens=256,
        temperature=0.2,
        top_p=0.85,
        top_k=30,
        chunk_size=512,
        chunk_overlap=25,
        retriever_k=3,
    ),
}
