from medical_assistant.config.personas import PERSONAS, PersonaConfig
from medical_assistant.config.settings import Settings
from medical_assistant.core.llm import get_llm, is_loaded
from medical_assistant.core.vector_store import get_vector_store
from medical_assistant.prompts.templates import SYSTEM_PROMPTS, USER_MESSAGE_TEMPLATE
from medical_assistant.rag.retriever import get_retriever

_NO_CONTEXT_REPLY = (
    "I could not find relevant information in the medical reference to answer your question. "
    "Please consult a qualified healthcare professional."
)


class RAGPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if settings.llm_backend == "llama_cpp":
            self.llm = get_llm(settings)
        else:
            self.llm = None

    def is_ready(self) -> bool:
        if self.settings.llm_backend == "groq":
            from medical_assistant.core.groq_llm import is_ready as groq_ready
            return groq_ready()
        return is_loaded()

    def run_query(self, question: str, persona: str) -> dict:
        if persona not in PERSONAS:
            raise ValueError(
                f"Unknown persona '{persona}'. Valid options: {list(PERSONAS.keys())}"
            )

        config = PERSONAS[persona]
        store = get_vector_store(persona, self.settings)
        retriever = get_retriever(store, persona)

        docs = retriever.invoke(question)
        if not docs:
            return {"answer": _NO_CONTEXT_REPLY, "persona": persona, "sources": []}

        context = "\n\n".join(doc.page_content for doc in docs)
        sources = sorted({doc.metadata.get("source", "Medical Reference") for doc in docs})
        system_prompt = SYSTEM_PROMPTS[persona]
        user_message = USER_MESSAGE_TEMPLATE.format(context=context, question=question)

        if self.settings.llm_backend == "groq":
            answer = self._groq_generate(system_prompt, user_message, config)
        else:
            answer = self._llama_generate(system_prompt, user_message, config)

        return {"answer": answer, "persona": persona, "sources": sources, "context": context}

    def _llama_generate(self, system_prompt: str, user_message: str, config: PersonaConfig) -> str:
        full_prompt = f"[INST]<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message}[/INST]"
        response = self.llm(
            full_prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            stop=["[INST]", "</s>"],
        )
        return response["choices"][0]["text"].strip()

    def _groq_generate(self, system_prompt: str, user_message: str, config: PersonaConfig) -> str:
        from medical_assistant.core.groq_llm import get_groq_client
        client = get_groq_client(self.settings)
        response = client.chat.completions.create(
            model=self.settings.groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )
        return response.choices[0].message.content.strip()
