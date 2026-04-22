from pathlib import Path

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from medical_assistant.config.settings import Settings

_llm_instance: Llama | None = None


def get_llm(settings: Settings) -> Llama:
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    model_path = Path(settings.model_path)
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=settings.model_repo_id,
            filename=settings.model_filename,
            local_dir=str(model_path.parent),
        )

    _llm_instance = Llama(
        model_path=str(model_path),
        n_ctx=settings.n_ctx,
        n_batch=settings.n_batch,
        n_gpu_layers=settings.n_gpu_layers,
        verbose=False,
    )
    return _llm_instance


def is_loaded() -> bool:
    return _llm_instance is not None
