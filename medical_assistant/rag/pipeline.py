from medical_assistant.config.personas import PERSONAS
from medical_assistant.config.settings import Settings
from medical_assistant.core.llm import get_llm
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
        self.llm = get_llm(settings)

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
        full_prompt = f"[INST]<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message}[/INST]"

        response = self.llm(
            full_prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            stop=["[INST]", "</s>"],
        )

        answer = response["choices"][0]["text"].strip()
        return {"answer": answer, "persona": persona, "sources": sources, "context": context}
