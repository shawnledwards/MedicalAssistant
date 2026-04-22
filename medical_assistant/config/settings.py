from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Model
    model_repo_id: str = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    model_filename: str = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    model_path: str = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

    # GPU: set to 35 for full RTX 3070 offload, 0 for CPU-only (HF Spaces free tier)
    n_gpu_layers: int = 0
    n_ctx: int = 5000
    n_batch: int = 512

    # Embeddings
    embedding_model: str = "thenlper/gte-large"

    # Data
    pdf_path: str = "data/medical_diagnosis_manual.pdf"
    vector_db_path: str = "data/vectorstore"

    # Server
    host: str = "0.0.0.0"
    port: int = 7860

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
