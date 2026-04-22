from groq import Groq

from medical_assistant.config.settings import Settings

_client: Groq | None = None


def get_groq_client(settings: Settings) -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.groq_api_key)
    return _client


def is_ready() -> bool:
    return _client is not None
